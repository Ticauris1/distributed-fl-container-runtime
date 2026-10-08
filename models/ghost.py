"""Ghost-module model replacement helpers."""

import math
import torch #type: ignore
import torch.nn as nn #type: ignore


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, inputs):
        primary = self.primary_conv(inputs)
        cheap = self.cheap_operation(primary)
        return torch.cat([primary, cheap], dim=1)[:, : self.oup, :, :]


def patch_model_with_ghost(model, model_name=None):
    if model_name is not None and "inception" in model_name.lower():
        print(f"Skipping Ghost patch for {model_name}")
        return model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return model
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            setattr(model, name, GhostModule(module.in_channels, module.out_channels, 3, stride=module.stride[0]).to(device))
        else:
            patch_model_with_ghost(module, model_name=model_name)
    return model
