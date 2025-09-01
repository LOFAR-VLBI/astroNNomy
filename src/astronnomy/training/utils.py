import os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from functools import partial, lru_cache
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from torch.nn.functional import interpolate
import torcheval.metrics.functional as tef
import random
import wandb
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt


class Rotate90Transform:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles)
        return v2.functional.rotate(x, int(angle), InterpolationMode.BILINEAR)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class RotateAndCrop:
    def __init__(self, rotate=True):
        self.rotate = rotate
        self.max_crop_factor = np.sqrt(2)

    def __call__(self, x):
        angle = 0
        if self.rotate:
            angle = np.random.uniform(-180, 180)
            x = v2.functional.rotate(x, angle, InterpolationMode.BILINEAR)

        angle_rad = np.deg2rad(angle)
        *_, h, w = x.shape
        # Compute the factor with which the image needs to be cropped to perfectly fit the largest possible square
        min_crop_factor = abs(np.cos(angle_rad)) + abs(np.sin(angle_rad))
        crop_size = int(
            min(h, w) / np.random.uniform(min_crop_factor, self.max_crop_factor)
        )

        return v2.functional.center_crop(x, crop_size)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class ResizeAndNoise:
    def __init__(self, resize=0):
        self.resize = resize

    def __call__(self, x):
        *_, h, w = x.shape
        resize_factor = self.resize / h
        sigma = 0.5 * resize_factor
        x = x + torch.randn_like(x) * sigma
        x = interpolate(
            x,
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False,
        )
        return x

    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class RandomResizeAndNoise:
    def __init__(self, resize_min=0, resize_max=0):
        self.sizes = list(range(resize_min, resize_max + 1, 56))

    def __call__(self, x):
        *_, h, w = x.shape
        size = np.random.choice(self.sizes)
        resize_factor = size / h
        sigma = 0.5 * resize_factor
        x = x + torch.randn_like(x) * sigma
        x = interpolate(
            x,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        return x

    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


def get_logging_dir(logging_root: str, /, **kwargs):
    # As version string, prefer $SLURM_ARRAY_JOB_ID, then $SLURM_JOB_ID, then 0.
    version = int(os.getenv("SLURM_ARRAY_JOB_ID", os.getenv("SLURM_JOB_ID", 0)))
    version_appendix = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    while True:
        version_dir = "__".join(
            (
                f"version_{version}_{version_appendix}",
                *(f"{k}_{v}" for k, v in kwargs.items()),
            )
        )

        logging_dir = Path(logging_root) / version_dir

        if not logging_dir.exists():
            break
        version_appendix += 1

    return str(logging_dir.resolve())


def get_tensorboard_logger(logging_dir):
    writer = SummaryWriter(log_dir=str(logging_dir))

    # writer.add_hparams()

    return writer


def merge_metrics(suffix, **kwargs):
    return {f"{k}/{suffix}": v for k, v in kwargs.items()}


def write_metrics(writer, metrics: dict, global_step: int):
    for metric_name, value in metrics.items():
        if isinstance(value, tuple):
            probs, labels = value
            writer_fn = partial(
                writer.add_pr_curve,
                labels=labels,
                predictions=probs,
            )
        else:
            writer_fn = partial(writer.add_scalar, scalar_value=value)

        writer_fn(tag=metric_name, global_step=global_step)


def wb_write_metrics(metrics: dict, global_step: int):
    additional_metrics = {}
    for metric_name, value in metrics.items():
        if "pr_curve" in metric_name and isinstance(value, tuple):
            preds, targets = value
            preds = preds.cpu()
            targets = targets.cpu()
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_predictions(
                targets.cpu(),
                preds.cpu(),
                ax=ax,
                name="stop",
            )
            # Log the matplotlib figure as a WandB image
            metrics[metric_name] = wandb.Image(fig)
            plt.close(fig)  # Close the figure to avoid memory leaks

    else:
        wandb.log(metrics | additional_metrics, step=global_step)


@torch.no_grad()
def log_metrics(
    loss,
    logits,
    targets,
    log_suffix,
    global_step,
    write_metrics_f,
    **kwargs,
):
    # 1 -  preds since 'stop' (0) is the positive class
    targets = 1 - targets
    probs = 1 - torch.sigmoid(logits)
    ap = tef.binary_auprc(probs, targets)

    metrics = merge_metrics(
        bce_loss=loss,
        au_pr_curve=ap,
        **(
            {"pr_curve": (probs.to(torch.float32), targets.to(torch.float32))}
            if "training" not in log_suffix
            else {}
        ),
        suffix=log_suffix,
        **kwargs,
    )

    write_metrics_f(metrics=metrics, global_step=global_step)


def label_smoother(
    labels: torch.tensor, smoothing_factor: float = 0.1, stochastic: bool = True
):
    smoothing_factor = smoothing_factor - (
        torch.rand_like(labels) * smoothing_factor * stochastic
    )
    smoothed_label = (1 - smoothing_factor) * labels + 0.5 * smoothing_factor
    return smoothed_label


@lru_cache(maxsize=2)
def get_transforms(
    transform_group: str = "C1",
    crop: bool = False,
    resize_min: int = 0,
    resize_max: int = 0,
    resize_val: int = 560,
    mean: float = 0,
    std: float = 1,
    val: bool = False,
    **kwargs,
):

    if val:
        resize_min = resize_val if resize_val else resize_max
        resize_max = resize_val if resize_val else resize_max

    assert (
        resize_min <= resize_max
    ), "resize_min must be smaller or equal than resize_max"

    transform_group = "C1" if val else transform_group
    crop = True if "O" in transform_group else crop

    transforms = [v2.Normalize(mean=mean, std=std)]

    if transform_group == "D4":
        transforms.append(Rotate90Transform())
    elif transform_group == "O2" or crop:
        transforms.append(RotateAndCrop(rotate=transform_group == "O2"))

    if resize_min and resize_max:
        transforms.append(RandomResizeAndNoise(resize_min, resize_max))

    if transform_group != "C1" and not val:
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
    return v2.Compose(transforms)


def set_seed(seed):
    os.environ["TRAIN_SEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_optimizer(parameters: list[torch.Tensor], lr: float, **kwargs):
    return torch.optim.AdamW(parameters, lr=lr, **kwargs)


def load_checkpoint(ckpt_path, device="cuda"):
    if os.path.isfile(ckpt_path):
        ckpt_dict = torch.load(ckpt_path, weights_only=False, map_location=device)
    else:
        files = os.listdir(ckpt_path)
        possible_checkpoints = list(filter(lambda x: x.endswith(".pth"), files))
        if len(possible_checkpoints) == 0:
            raise ValueError(
                f"No checkpoint file found in the given checkpoint directory: {ckpt_path}"
            )
        elif len(possible_checkpoints) != 1:
            raise ValueError(
                f"Too many checkpoint files in the given checkpoint directory. Please specify the model you want to load directly."
            )
        ckpt_path = f"{ckpt_path}/{possible_checkpoints[0]}"
        ckpt_dict = torch.load(ckpt_path, weights_only=False, map_location=device)

    config = ckpt_dict["config"]

    model = ckpt_dict["model"](**config.model).to(device)

    model.load_state_dict(ckpt_dict["model_state_dict"], strict=False)

    optim = ckpt_dict.get("optimizer", torch.optim.AdamW)(
        params=[param for param in model.parameters() if param.requires_grad],
        **config.optimizer,
    )
    optim.load_state_dict(ckpt_dict["optimizer_state_dict"])

    return {"model": model, "optim": optim, "config": config}


def save_checkpoint(logging_dir, model, optimizer, global_step, config, **kwargs):
    os.makedirs(logging_dir, exist_ok=True)
    old_checkpoints = Path(logging_dir).glob("*.pth")
    for old_checkpoint in old_checkpoints:
        Path.unlink(old_checkpoint)

    checkpoint_path = logging_dir + f"/ckpt_step={global_step}.pth"

    torch.save(
        {
            "model": type(model),
            "model_state_dict": model.finetuned_state_dict(),
            "optimizer": type(optimizer),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": global_step,
            "config": config,
            **kwargs,
        },
        f=checkpoint_path,
    )

    return checkpoint_path


if __name__ == "__main__":

    thing = RandomResizeAndNoise(534, 899)
    print(thing)
