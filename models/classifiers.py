"""Neural-network classifier wrappers."""

import timm #type: ignore
import torch #type: ignore
import torch.nn as nn #type: ignore


class ImageClassifier(nn.Module):
    """Wrap a timm backbone with global pooling and a linear classifier."""

    def __init__(self, model_name="resnet50", num_classes=10, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        from .factory import get_backbone_out_features

        in_features = get_backbone_out_features(self.backbone, model_name)
        self.head = nn.Linear(in_features, num_classes)
        print(f"✅ Created {model_name} with a head accepting {in_features} features.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        if features.ndim == 4:
            features = self.pool(features).flatten(1)
        elif features.ndim != 2:
            raise ValueError(
                f"Unexpected backbone output shape during forward: {tuple(features.shape)}"
            )
        return self.head(features)


class VGG16Classifier(nn.Module):
    """VGG16 conv features followed by global pooling and a linear head."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vgg16", pretrained=pretrained, features_only=True, out_indices=[-1]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_channels = self.backbone.feature_info.channels()[-1]
        self.head = nn.Linear(in_channels, num_classes)
        print(
            "✅ Created vgg16 (features_only) with a head accepting "
            f"{in_channels} features."
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)[-1]
        return self.head(self.pool(features).flatten(1))
