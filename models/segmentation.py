"""Configuration-driven segmentation-model construction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


SEGMENTATION_MODEL_BUILDERS = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "manet": "MAnet",
    "linknet": "Linknet",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "pan": "PAN",
    "deeplabv3": "DeepLabV3",
    "deeplabv3plus": "DeepLabV3Plus",
}


class ChannelAttention(nn.Module):
    """CBAM channel attention using shared weights for average/max pooling."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(1, int(channels) // int(reduction))
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(int(channels), hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, int(channels), kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        weights = self.shared_mlp(self.average_pool(inputs))
        weights = weights + self.shared_mlp(self.maximum_pool(inputs))
        return inputs * self.activation(weights)


class SpatialAttention(nn.Module):
    """CBAM spatial attention over channel-average and channel-maximum maps."""

    def __init__(self, kernel_size=7):
        super().__init__()
        if int(kernel_size) not in {3, 7}:
            raise ValueError("CBAM spatial kernel size must be 3 or 7")
        padding = int(kernel_size) // 2
        self.convolution = nn.Conv2d(
            2, 1, kernel_size=int(kernel_size), padding=padding, bias=False
        )
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        average = torch.mean(inputs, dim=1, keepdim=True)
        maximum = torch.amax(inputs, dim=1, keepdim=True)
        weights = self.activation(self.convolution(torch.cat([average, maximum], dim=1)))
        return inputs * weights


class CBAM(nn.Module):
    """Sequential channel and spatial attention from the CBAM design."""

    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, inputs):
        return self.spatial_attention(self.channel_attention(inputs))


class EncoderWithBottleneckCBAM(nn.Module):
    """Apply CBAM only to the deepest SMP encoder feature map."""

    def __init__(self, encoder, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.encoder = encoder
        self.cbam = CBAM(
            encoder.out_channels[-1], reduction=reduction,
            spatial_kernel_size=spatial_kernel_size,
        )
        self.out_channels = encoder.out_channels
        self.output_stride = encoder.output_stride

    def forward(self, inputs):
        features = list(self.encoder(inputs))
        features[-1] = self.cbam(features[-1])
        return features


class PowerOfTwoFeaturePyramidEncoder(nn.Module):
    """Align irregular encoder feature maps for U-Net-style skip connections."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.out_channels = encoder.out_channels
        self.output_stride = encoder.output_stride

    def forward(self, inputs):
        features = list(self.encoder(inputs))
        input_height, input_width = inputs.shape[-2:]
        aligned = [features[0]]
        for level, feature in enumerate(features[1:], start=1):
            target_size = (
                max(1, input_height // (2 ** level)),
                max(1, input_width // (2 ** level)),
            )
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            aligned.append(feature)
        return aligned


class ContiguousPABBlock(nn.Module):
    """MAnet PAB equivalent that is safe for non-contiguous MPS tensors."""

    def __init__(self, source):
        super().__init__()
        self.pab_channels = source.pab_channels
        self.in_channels = source.in_channels
        self.top_conv = source.top_conv
        self.center_conv = source.center_conv
        self.bottom_conv = source.bottom_conv
        self.map_softmax = source.map_softmax
        self.out_conv = source.out_conv

    def forward(self, inputs):
        inputs = inputs.contiguous()
        batch_size, _, height, width = inputs.shape
        pixel_count = height * width
        top = self.top_conv(inputs).contiguous().reshape(
            batch_size, self.pab_channels, pixel_count
        )
        center = self.center_conv(inputs).contiguous().reshape(
            batch_size, self.pab_channels, pixel_count
        ).transpose(1, 2).contiguous()
        bottom = self.bottom_conv(inputs).contiguous().reshape(
            batch_size, self.in_channels, pixel_count
        ).transpose(1, 2).contiguous()
        spatial_map = torch.matmul(center, top).contiguous()
        spatial_map = self.map_softmax(
            spatial_map.reshape(batch_size, pixel_count * pixel_count)
        ).contiguous()
        spatial_map = spatial_map.reshape(
            batch_size, height * width, height * width
        ).contiguous()
        spatial_map = torch.matmul(spatial_map, bottom).contiguous()
        spatial_map = spatial_map.reshape(
            batch_size, self.in_channels, height, width
        ).contiguous()
        return self.out_conv((inputs + spatial_map).contiguous()).contiguous()


def normalize_segmentation_model_name(model_name):
    return str(model_name).strip().lower().replace("-", "").replace("_", "").replace("+", "plus")


def normalize_segmentation_encoder_name(encoder_name):
    """Translate project backbone names to segmentation-models-pytorch names."""
    normalized = str(encoder_name).strip().lower()
    # The classification factory uses the LW- prefix to request Ghost-module
    # replacement. SMP owns its encoder construction, so use the corresponding
    # base encoder here; Ghost conversion would require a separate adapter.
    normalized = normalized.removeprefix("lw-").strip()
    universal_timm_aliases = {
        "mobilenetv2_100": "tu-mobilenetv2_100",
        "inception_v3": "tu-inception_v3",
        "inceptionv3": "tu-inception_v3",
    }
    normalized = universal_timm_aliases.get(normalized, normalized)
    if normalized.startswith("efficientnet_"):
        normalized = normalized.replace("efficientnet_", "efficientnet-", 1)
    return normalized


def make_segmentation_model(
    model_name,
    device,
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    attention="none",
    cbam_reduction=16,
    cbam_spatial_kernel_size=7,
):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise ImportError(
            "Install segmentation-models-pytorch in the active environment before "
            "running segmentation experiments."
        ) from error
    normalized_name = normalize_segmentation_model_name(model_name)
    class_name = SEGMENTATION_MODEL_BUILDERS.get(normalized_name)
    if class_name is None:
        supported = ", ".join(sorted(SEGMENTATION_MODEL_BUILDERS))
        raise ValueError(
            f"Unsupported segmentation model {model_name!r}. Supported: {supported}"
        )
    encoder_name = normalize_segmentation_encoder_name(encoder_name)
    available_encoders = set(smp.encoders.get_encoder_names())
    # SMP's universal timm encoders use the ``tu-`` prefix and are resolved
    # dynamically by get_encoder(), so they are not included in this registry.
    if encoder_name not in available_encoders and not encoder_name.startswith("tu-"):
        raise ValueError(
            f"Backbone {encoder_name!r} is not available as an SMP segmentation "
            "encoder. Choose a compatible entry in config.DEEP_MODELS."
        )
    model_class = getattr(smp, class_name)
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=int(in_channels),
        classes=int(classes),
        activation=None,
    )
    if encoder_name == "tu-inception_v3":
        model.encoder = PowerOfTwoFeaturePyramidEncoder(model.encoder)
    # SMP's MAnet PAB uses Tensor.view after a matrix multiplication. MPS can
    # return that result with non-contiguous strides, causing backward() to
    # fail. This block is mathematically identical and uses reshape instead.
    if normalized_name == "manet":
        model.decoder.center = ContiguousPABBlock(model.decoder.center)
    normalized_attention = str(attention).strip().lower().replace("-", "_")
    if normalized_attention in {"", "none"}:
        pass
    elif normalized_attention == "bottleneck_cbam":
        model.encoder = EncoderWithBottleneckCBAM(
            model.encoder,
            reduction=int(cbam_reduction),
            spatial_kernel_size=int(cbam_spatial_kernel_size),
        )
    else:
        raise ValueError(
            f"Unsupported segmentation attention {attention!r}. "
            "Supported: none, bottleneck_cbam"
        )
    return model.to(device)


# Backward-compatible convenience wrapper for earlier pilot code.
def make_unetplusplus(device, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    return make_segmentation_model(
        "unetplusplus", device, encoder_name, encoder_weights, in_channels, classes
    )
