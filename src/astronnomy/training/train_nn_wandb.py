import argparse
from argparse import Namespace
import yaml
import wandb
import warnings
from functools import partial
from pathlib import Path
import torch
from torch import nn, binary_cross_entropy_with_logits
from torchvision.transforms import v2
from tqdm import tqdm

from .models import ImagenetTransferLearning
from .utils import (
    get_logging_dir,
    get_tensorboard_logger,
    log_metrics,
    wb_write_metrics,
    label_smoother,
    get_transforms,
    set_seed,
    save_checkpoint,
    get_optimizer,
    load_checkpoint,
)
from .data import get_dataloaders, prepare_data
from .train_nn_yaml import *


def setup_wandb(config, model):
    wandb.login()
    wandb.init(config=config, **config.logging["wandb"])


def main(
    config: Namespace,
):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")

    model: nn.Module = ImagenetTransferLearning(**config.model)

    logging_dir = get_logging_dir(
        str(Path.cwd() / config.logging["log_dir"]),
        **config.model,
    )

    # noinspection PyArgumentList
    model.to(device=device, memory_format=torch.channels_last)

    optimizer = get_optimizer(
        [param for param in model.parameters() if param.requires_grad],
        **config.optimizer,
    )

    train_dataloader, val_dataloader = get_dataloaders(**config.dataloader)

    mean, std = train_dataloader.dataset.compute_statistics(
        config.data_transforms["normalize"]
    )

    config.data_transforms["mean"] = mean.to(torch.float32)
    config.data_transforms["std"] = std.to(torch.float32)
    config.logging["full_log_dir"] = logging_dir

    logging_interval = config.logging["log_frequency"]

    train_step_f, val_step_f = (
        partial(
            step_f,
            prepare_data_f=partial(
                prepare_data,
                device=device,
                transform=get_transforms(
                    **config.data_transforms,
                    val=bool(val),
                ),
            ),
            metrics_logger=partial(log_metrics, write_metrics_f=wb_write_metrics),
        )
        for val, step_f in enumerate([train_step, val_step])
    )

    train_step_f = partial(
        train_step_f,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        logging_interval=logging_interval,
        smoothing_fn=partial(
            label_smoother,
            stochastic=config.training["stochastic_smoothing"],
            smoothing_factor=config.training["label_smoothing"],
        ),
    )
    val_step_f = partial(val_step_f, val_dataloader=val_dataloader)

    checkpoint_saver = partial(
        save_checkpoint,
        logging_dir=logging_dir,
        model=model,
        optimizer=optimizer,
        config=config,
    )

    best_val_loss = torch.inf
    global_step = 0  # make it a tensor so we can do in-place edits

    best_results = {}

    setup_wandb(config, model)

    n_epochs = config.training["epochs"]
    for epoch in range(n_epochs):

        global_step = train_step_f(global_step=global_step, model=model)
        val_loss, logits, targets = val_step_f(global_step=global_step, model=model)
        if val_loss < best_val_loss:
            best_results["logits"] = logits.clone()
            best_results["targets"] = targets.clone()
            checkpoint_path = checkpoint_saver(global_step=global_step)
            best_val_loss = val_loss

        with torch.no_grad():
            log_metrics(
                loss=best_val_loss,
                logits=best_results["logits"],
                targets=best_results["targets"],
                global_step=global_step,
                log_suffix="validation_best",
                write_metrics_f=wb_write_metrics,
            )

    print(f"Validation runs:")
    with torch.no_grad():
        # checkpoint_path = ""
        model = load_checkpoint(checkpoint_path)["model"]
        augment_transforms, val_transforms = get_augment_transforms(
            **config.data_transforms,
        )
        prepare_data_dict = {
            key: partial(
                prepare_data,
                device=device,
                transform=transform,
            )
            for key, transform in augment_transforms.items()
        }

        aug_val_step(
            model=model,
            val_dataloader=val_dataloader,
            global_step=global_step,
            metrics_logger=partial(log_metrics, write_metrics_f=wb_write_metrics),
            aug_prep_dict=prepare_data_dict,
            val_prep_f=partial(prepare_data, device=device, transform=val_transforms),
        )

    wandb.run.finish()


@torch.no_grad()
def aug_val_step(
    model, val_dataloader, global_step, metrics_logger, aug_prep_dict, val_prep_f
):
    val_losses, val_logits, val_targets, val_pred_errors, val_feature_error = (
        {key: [] for key in aug_prep_dict},
        {key: [] for key in aug_prep_dict},
        {key: [] for key in aug_prep_dict},
        {key: [] for key in aug_prep_dict},
        {key: [] for key in aug_prep_dict},
    )

    model.eval()
    for i, (data, labels) in tqdm(
        enumerate(val_dataloader),
        desc="Validate augmentation",
        total=len(val_dataloader),
    ):
        # print("validation start")
        val_data, _ = val_prep_f(data, labels)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base_features = model.feature_extractor(model.lift(val_data))
            base_preds = torch.sigmoid(model.classifier(base_features).flatten())
        for key, prep_f in aug_prep_dict.items():
            aug_data, aug_labels = prep_f(data, labels)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.feature_extractor(model.lift(aug_data))
                feature_error = torch.nn.functional.mse_loss(
                    features.clone(), base_features, reduction="none"
                ).mean(dim=1)
                logits = model.classifier(features).flatten()
                loss = binary_cross_entropy_with_logits(
                    logits,
                    aug_labels,
                    pos_weight=torch.as_tensor(val_dataloader.dataset.label_ratio),
                )
            pred_error = (torch.sigmoid(logits) - (base_preds)).abs()
            val_losses[key].append(loss)
            val_logits[key].append(logits.clone())
            val_targets[key].append(aug_labels)
            val_pred_errors[key].append(pred_error)
            val_feature_error[key].append(feature_error)
    for key in aug_prep_dict:
        losses, logits, targets, errors, feature_errors = map(
            torch.concatenate,
            (
                val_losses[key],
                val_logits[key],
                val_targets[key],
                val_pred_errors[key],
                val_feature_error[key],
            ),
        )

        mean_loss = losses.mean()
        mean_error = errors.mean()
        feature_error = feature_errors.mean()
        metrics_logger(
            loss=mean_loss,
            logits=logits,
            targets=targets,
            symmetry_error=mean_error,
            feature_error=feature_error,
            global_step=global_step,
            log_suffix=f"aug_{key}",
        )


def get_augment_transforms(
    transform_group: str = "C1",
    crop: bool = False,
    resize_min: int = 0,
    resize_max: int = 0,
    resize_val: int = 560,
    mean: float = 0,
    std: float = 1,
    **kwargs,
):

    transforms = {}
    val_transforms = get_transforms(
        transform_group=transform_group,
        crop=crop,
        resize_val=resize_val,
        resize_min=resize_min,
        resize_max=resize_max,
        mean=mean,
        std=std,
        val=True,
    )

    for group, crop_aug in [
        ("C1", False),
        ("C1", True),
        ("D4", False),
        ("D4", True),
        ("O2", True),
    ]:

        transforms[f"group={group}, crop={crop_aug}"] = get_transforms(
            transform_group=group,
            crop=crop_aug,
            resize_min=resize_val,
            resize_max=resize_val,
            mean=mean,
            std=std,
            val=False,
        )

    for resize in range(224, 1344 + 1, 224):
        transforms[f"resize={resize}"] = get_transforms(
            transform_group="C1",
            crop=crop,
            resize_min=resize,
            resize_max=resize,
            mean=mean,
            std=std,
            val=False,
        )

    return transforms, val_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)

    print(config)

    sanity_check_config(config)

    print(config)

    if config.reproducibility["seed"] is not None:
        set_seed(config.reproducibility["seed"])

    main(config)
