"""Deep-model construction and transform helpers."""

from typing import Tuple
import timm #type: ignore
import torch #type: ignore
import torch.nn as nn #type: ignore
from torchvision import transforms #type: ignore

from .classifiers import ImageClassifier, VGG16Classifier
from .ghost import patch_model_with_ghost


def resolve_timm_input_hw(model_name: str) -> Tuple[int, int]:
    actual_name, _ = normalize_model_name(model_name)
    if actual_name == "paper_lightweight":
        return 224, 224
    try:
        config = timm.models.resolve_pretrained_cfg(actual_name)
        size = config.get("input_size", None)
        if size and len(size) == 3:
            return int(size[-2]), int(size[-1])
    except Exception:
        pass
    try:
        config = timm.models.get_pretrained_cfg(actual_name)
        size = getattr(config, "input_size", None)
        if size and len(size) == 3:
            return int(size[-2]), int(size[-1])
    except Exception:
        pass
    return 224, 224


def get_backbone_out_features(model: nn.Module, model_name: str) -> int:
    num_features = getattr(model, "num_features", None)
    if isinstance(num_features, int) and num_features > 0:
        return num_features
    height, width = resolve_timm_input_hw(model_name)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, height, width))
        if output.ndim == 4:
            output = nn.AdaptiveAvgPool2d(1)(output).flatten(1)
        elif output.ndim != 2:
            raise ValueError(f"Unexpected backbone output shape: {tuple(output.shape)}")
        return int(output.shape[1])

def normalize_model_name(model_name: str) -> tuple[str, bool]:
    normalized = model_name.strip().lower()
    use_ghost = normalized.startswith("lw-")
    actual_name = normalized.removeprefix("lw-").strip()
    return actual_name, use_ghost

def make_dl_model(model_name, num_classes, device, pretrained=True):
    actual_name, use_ghost = normalize_model_name(model_name)

    if actual_name == "vgg16":
        model = VGG16Classifier(num_classes, pretrained).to(device)

    elif actual_name == "paper_lightweight":
        from .lightweight import LightweightCNN
        model = LightweightCNN(num_classes).to(device)

    else:
        model = ImageClassifier(
            actual_name,
            num_classes,
            pretrained,
        ).to(device)

    return patch_model_with_ghost(model, model_name) if use_ghost else model


def get_transforms(model_name):
    clean_name, _ = normalize_model_name(model_name)

    default_mean = (0.485, 0.456, 0.406)
    default_std = (0.229, 0.224, 0.225)

    if clean_name == "paper_lightweight":
        image_size = 224
        mean, std = default_mean, default_std
        print("⚡ paper_lightweight detected: using 224x224")

    elif "inception" in clean_name:
        image_size = 299
        mean, std = default_mean, default_std
        print("🚀 Inception detected: using 299x299")

    else:
        try:
            pretrained_cfg = timm.models.resolve_pretrained_cfg(clean_name)
            input_size = pretrained_cfg.input_size

            image_size = (
                int(input_size[-2]),
                int(input_size[-1]),
            )
            mean = pretrained_cfg.mean
            std = pretrained_cfg.std

        except Exception as error:
            image_size = (224, 224)
            mean, std = default_mean, default_std
            print(
                f"Warning: Could not resolve timm configuration for "
                f"{clean_name!r}: {error}. Using 224x224."
            )

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transform, transform


_resolve_timm_input_hw = resolve_timm_input_hw
_get_backbone_out_features = get_backbone_out_features
