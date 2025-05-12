from torch import nn, binary_cross_entropy_with_logits
import torch
from functools import partial
from torchvision import models

from .dino_model import DINOV2FeatureExtractor


class ImagenetTransferLearning(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        dropout_p: float = 0.25,
        use_compile: bool = True,
        lift: str = "stack",
        use_lora: bool = False,
        lora_alpha: float = 16.0,
        lora_rank: int = 16,
        tune_register_tokens: bool = False,
        tune_cls_token: bool = False,
        pos_embed: str = "pre-trained",
        **kwargs,
    ):
        super().__init__()

        get_classifier_f = partial(
            get_classifier, dropout_p=dropout_p, num_target_classes=1
        )

        # For saving in the state dict
        self.kwargs = {
            "model_name": model_name,
            "dropout_p": dropout_p,
            "use_compile": use_compile,
            "lift": lift,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "tune_register_tokens": tune_register_tokens,
            "tune_cls_token": tune_cls_token,
            "pos_embed": pos_embed,
        }

        if lift == "stack":
            self.lift = partial(torch.repeat_interleave, repeats=3, dim=1)
        elif lift == "conv":
            self.lift = nn.Conv2d(1, 3, 1)
        elif lift == "reinit_first":
            self.lift = nn.Identity()

        if model_name == "vit_l_16":
            assert tune_cls_token, "tune_cls_token must be True for ViT models"
            self.feature_extractor, num_features = init_vit(model_name)
            self.feature_extractor.eval()

            self.feature_extractor.heads = nn.Identity()

        elif "dinov2" in model_name:
            self.feature_extractor, num_features = init_dino(
                model_name,
                use_lora=use_lora,
                alpha=lora_alpha,
                rank=lora_rank,
            )
            if "zeros" in pos_embed:
                self.feature_extractor.encoder.pos_embed[:, 1:, :] = torch.zeros_like(
                    self.feature_extractor.encoder.pos_embed[:, 1:, :]
                )

            if "fine-tune" in pos_embed:
                self.feature_extractor.encoder.pos_embed.requires_grad_(True)
            if tune_register_tokens:
                self.feature_extractor.encoder.register_tokens.requires_grad_(True)
            if tune_cls_token:
                self.feature_extractor.encoder.cls_token.requires_grad_(True)

            if use_lora:
                self.feature_extractor.train()
            else:
                self.feature_extractor.eval()

        else:
            self.feature_extractor, num_features = init_cnn(name=model_name, lift=lift)
            self.feature_extractor.eval()

        self.classifier = get_classifier_f(n_features=num_features)

        if use_compile:
            self.forward = torch.compile(model=self.forward, mode="reduce-overhead")

    def forward(self, x):
        x = self.lift(x)
        features = self.feature_extractor(x)
        return self.classifier(features)

    def step(self, inputs, targets, ratio=1):
        logits = self(inputs).flatten()

        loss = binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.as_tensor(ratio)
        )

        return logits, loss

    def eval(self):
        self.classifier.eval()
        if "dinov2" in self.kwargs["model_name"] and self.kwargs["use_lora"]:
            self.feature_extractor.eval()

    def train(self):
        self.classifier.train()
        if "dinov2" in self.kwargs["model_name"] and self.kwargs["use_lora"]:
            self.feature_extractor.train()

    def finetuned_state_dict(self):
        pretrained_state_dict = {}
        classifier_state_dict = self.classifier.state_dict()
        for key, value in classifier_state_dict.items():
            pretrained_state_dict[f"classifier.{key}"] = value

        if "dinov2" in self.kwargs["model_name"]:
            if self.kwargs["use_lora"]:
                for (
                    key,
                    value,
                ) in self.feature_extractor.encoder.blocks.state_dict().items():
                    if "qkv.linear" in key:
                        pretrained_state_dict[
                            f"feature_extractor.encoder.blocks{key}"
                        ] = value

            if self.kwargs["tune_cls_token"]:
                pretrained_state_dict[f"feature_extractor.encoder.cls_token"] = (
                    self.feature_extractor.encoder.state_dict()["cls_token"]
                )

            if self.kwargs["tune_register_tokens"]:
                pretrained_state_dict[f"feature_extractor.encoder.register_tokens"] = (
                    self.feature_extractor.encoder.state_dict()["register_tokens"]
                )

            if "fine-tune" in self.kwargs["pos_embed"]:
                pretrained_state_dict[f"feature_extractor.encoder.pos_embed"] = (
                    self.feature_extractor.encoder.state_dict()["pos_embed"]
                )

        elif self.kwargs["model_name"] == "vit_l_16":
            pretrained_state_dict["feature_extractor.class_token"] = (
                self.feature_extractor.state_dict()["class_token"]
            )

        return pretrained_state_dict


def init_vit(model_name):
    assert model_name == "vit_l_16"

    backbone = models.vit_l_16(weights="ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1")
    for param in backbone.parameters():
        param.requires_grad_(False)

    backbone.class_token.requires_grad_(True)

    hidden_dim = backbone.heads[0].in_features

    del backbone.heads
    return backbone, hidden_dim


def init_dino(model_name, use_lora, rank=16, alpha=32):

    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    hidden_dim = backbone.cls_token.shape[-1]
    classifier = nn.Identity()

    dino_lora = DINOV2FeatureExtractor(
        encoder=backbone,
        decoder=classifier,
        r=rank,
        use_lora=use_lora,
        alpha=alpha,
    )

    return dino_lora, hidden_dim


def init_cnn(name: str, lift="stack"):
    # use partial to prevent loading all models at once
    model_map = {
        "resnet50": partial(models.resnet50, weights="DEFAULT"),
        "resnet152": partial(models.resnet152, weights="DEFAULT"),
        "resnext50_32x4d": partial(models.resnext50_32x4d, weights="DEFAULT"),
        "resnext101_64x4d": partial(models.resnext101_64x4d, weights="DEFAULT"),
        "efficientnet_v2_l": partial(models.efficientnet_v2_l, weights="DEFAULT"),
    }

    backbone = model_map[name]()

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    for param in feature_extractor.parameters():
        param.requires_grad_(False)

    if lift == "reinit_first":
        if name in ("resnet50", "resnet152", "resnext50_32x4d", "resnext101_64x4d"):
            conv = feature_extractor[0]
            new_conv = init_first_conv(conv)
            feature_extractor[0] = new_conv
            del conv
        elif name == "efficientnet_v2_l":
            conv = feature_extractor[0][0][0]
            new_conv = init_first_conv(conv)
            feature_extractor[0][0][0] = new_conv
            del conv

    num_out_features = (
        backbone.fc
        if name in ("resnet50", "resnet152", "resnext50_32x4d", "resnext101_64x4d")
        else backbone.classifier[-1]  # efficientnet
    ).in_features
    return feature_extractor, num_out_features


def init_first_conv(conv):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    bias = conv.bias
    out_channels = conv.out_channels
    return nn.Conv2d(
        in_channels=1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def get_classifier(dropout_p: float, n_features: int, num_target_classes: int):
    assert 0 <= dropout_p <= 1
    print(n_features)

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout1d(p=dropout_p),
        nn.Linear(n_features, n_features),
        nn.ReLU(),
        nn.Linear(n_features, num_target_classes),
    )

    return classifier
