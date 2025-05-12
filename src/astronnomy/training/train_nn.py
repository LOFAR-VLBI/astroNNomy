import argparse
from argparse import Namespace
import yaml
import warnings
from functools import partial
from pathlib import Path
import torch
from torch import nn, binary_cross_entropy_with_logits

from tqdm import tqdm

from .models import ImagenetTransferLearning
from .utils import (
    get_logging_dir,
    get_tensorboard_logger,
    log_metrics,
    write_metrics,
    label_smoother,
    get_transforms,
    set_seed,
    save_checkpoint,
    get_optimizer,
)
from .data import get_dataloaders, prepare_data


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

    writer = get_tensorboard_logger(logging_dir)

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

    config.data_transforms["mean"] = mean
    config.data_transforms["std"] = std
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
            metrics_logger=partial(
                log_metrics, write_metrics_f=partial(write_metrics, writer=writer)
            ),
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

    n_epochs = config.training["epochs"]
    for epoch in range(n_epochs):

        global_step = train_step_f(global_step=global_step, model=model)
        val_loss, logits, targets = val_step_f(global_step=global_step, model=model)
        if val_loss < best_val_loss:
            best_results["logits"] = logits.clone()
            best_results["targets"] = targets.clone()
            checkpoint_saver(global_step=global_step)
            best_val_loss = val_loss

        with torch.no_grad():
            log_metrics(
                loss=best_val_loss,
                logits=best_results["logits"],
                targets=best_results["targets"],
                global_step=global_step,
                log_suffix="validation_best",
                write_metrics_f=partial(write_metrics, writer=writer),
            )

    writer.flush()
    writer.close()


@torch.no_grad()
def val_step(model, val_dataloader, global_step, metrics_logger, prepare_data_f):
    val_losses, val_logits, val_targets = [], [], []

    model.eval()
    for i, (data, labels) in tqdm(
        enumerate(val_dataloader), desc="Validation", total=len(val_dataloader)
    ):
        # print("validation start")

        data, labels = prepare_data_f(data, labels)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(data).flatten()
            loss = binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=torch.as_tensor(val_dataloader.dataset.label_ratio),
            )
        val_losses.append(loss)
        val_logits.append(logits.clone())
        val_targets.append(labels)

    losses, logits, targets = map(
        torch.concatenate, (val_losses, val_logits, val_targets)
    )

    mean_loss = losses.mean()
    metrics_logger(
        loss=mean_loss,
        logits=logits,
        targets=targets,
        global_step=global_step,
        log_suffix="validation",
    )

    return mean_loss, logits, targets


def train_step(
    model,
    optimizer,
    train_dataloader,
    prepare_data_f,
    global_step,
    logging_interval,
    metrics_logger,
    smoothing_fn,
):
    # print("training")
    model.train()

    for i, (data, labels) in tqdm(
        enumerate(train_dataloader), desc="Training", total=len(train_dataloader)
    ):
        global_step += 1
        data, labels = prepare_data_f(data, labels)
        smoothed_label = smoothing_fn(labels)
        # data = augmentation(data)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model.step(
                data, smoothed_label, ratio=train_dataloader.dataset.label_ratio
            )
            mean_loss = loss.mean()

        mean_loss.backward()
        optimizer.step()
        if i % logging_interval == 0:
            with torch.no_grad():
                metrics_logger(
                    loss=mean_loss.detach(),
                    logits=logits.detach(),
                    targets=labels,
                    global_step=global_step,
                    log_suffix="training",
                )

    return global_step


def sanity_check_config(config):
    assert config.optimizer["lr"] >= 0
    assert config.dataloader["batch_size"] >= 0
    assert config.data_transforms["resize_val"] >= 0
    assert 0 <= config.model["dropout_p"] <= 1
    assert 0 <= config.training["label_smoothing"] <= 1
    assert not (
        (
            config.model["model_name"] == "vit_l_16"
            or config.model["model_name"] == "dino_v2"
        )
        and config.model["lift"] == "reinit_first"
    )
    # ViT always needs the input size to be 512x512
    if config.model["model_name"] == "vit_l_16" and (
        config.data_transforms["resize_min"] != 512
        or config.data_transforms["resize_max"] != 512
        or config.data_transforms["resize_val"] != 512
    ):
        print("Setting resize to 512 since vit_16_l is being used")
        config.data_transforms["resize_min"] = 512
        config.data_transforms["resize_max"] = 512
        config.data_transforms["resize_val"] = 512
    if (
        "dinov2" in config.model["model_name"]
        and config.data_transforms["resize_min"] == 0
        and config.data_transforms["resize_max"] == 0
    ):
        resize = 560
        print(f"\n#######\nSetting resize to {resize} \n######\n")
        config.data_transforms["resize_min"] = resize
        config.data_transforms["resize_max"] = resize

    assert (
        config.data_transforms["resize_max"] >= 0
        and config.data_transforms["resize_min"] >= 0
    ), "resize_min and resize_max must be non-negative"

    if (
        config.data_transforms["resize_max"] > 0
        or config.data_transforms["resize_min"] > 0
    ):
        assert (
            config.data_transforms["resize_max"] > 0
            and config.data_transforms["resize_min"] > 0
        ), "If one of resize_min or resize_max is positive, both must be positive"

    if config.model["use_lora"] and not "dinov2" in config.model["model_name"]:
        warnings.warn(
            "Warning: LoRA is only supported for Dino V2 models. Ignoring setting....\n",
            UserWarning,
        )
        config.model["use_lora"] = False

    if config.model["lora_alpha"] is None:
        config.model["lora_alpha"] = config.model["lora_rank"] * 2

    assert (
        config.data_transforms["resize_min"] % 56 == 0
        or config.model["model_name"] != "dino_v2"
    ), "resizing must be a multiple of 14 for Dino V2 models (8 for efficiency, 14 for patch size)"
    assert (
        config.data_transforms["resize_max"] % 56 == 0
        or config.model["model_name"] != "dino_v2"
    ), "resizing must be a multiple of 14 for Dino V2 models (8 for efficiency, 14 for patch size)"

    assert (
        config.data_transforms["resize_val"] % 56 == 0
        or config.model["model_name"] != "dino_v2"
    ), "resizing must be a multiple of 14 for Dino V2 models (8 for efficiency, 14 for patch size)"

    assert config.model["pos_embed"] in [
        "pre-trained",
        "fine-tune",
        "zeros",
        "zeros+fine-tune",
    ], "pos_embed must be one of 'pre-trained', 'fine-tune', 'zeros', or 'zeros+fine-tune'"

    assert config.data_transforms["transform_group"] in [
        "C1",
        "D4",
        "O2",
    ], "transform_group must be one of 'C1', 'D4', or 'O2'"


def load_config(config_path, overrides):
    def str_to_bool(s):
        if isinstance(s, str):
            s_lower = s.lower()
            if s_lower == "true":
                return True
            elif s_lower == "false":
                return False
        elif isinstance(s, int):
            return bool(s)
        raise ValueError(f"Invalid boolean string: {s}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    for key_val in overrides:
        keys, val = key_val.split("=")
        keys = keys.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        val_type = type(d.get(keys[-1], val))
        if val_type == bool:
            val = str_to_bool(val)
        else:
            val = val_type(val)
        d[keys[-1]] = val  # cast to correct type
    return Namespace(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    print(args)

    config = load_config(args.config, args.overrides)

    print(config)

    sanity_check_config(config)

    if config.reproducibility["seed"] is not None:
        set_seed(config.reproducibility["seed"])

    main(config)
