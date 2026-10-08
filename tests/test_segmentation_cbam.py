import torch

from models.segmentation import CBAM, EncoderWithBottleneckCBAM


class _TinyEncoder(torch.nn.Module):
    out_channels = (3, 8, 16)
    output_stride = 4

    def forward(self, inputs):
        feature_1 = torch.nn.functional.avg_pool2d(inputs, 2)
        feature_1 = feature_1.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1)
        feature_2 = torch.nn.functional.avg_pool2d(feature_1, 2)
        feature_2 = torch.cat([feature_2, feature_2], dim=1)
        return [inputs, feature_1, feature_2]


def test_cbam_preserves_feature_shape_and_backpropagates():
    module = CBAM(channels=16, reduction=4, spatial_kernel_size=7)
    inputs = torch.randn(2, 16, 8, 8, requires_grad=True)
    outputs = module(inputs)
    assert outputs.shape == inputs.shape
    outputs.mean().backward()
    assert inputs.grad is not None


def test_encoder_wrapper_only_replaces_deepest_feature():
    encoder = EncoderWithBottleneckCBAM(_TinyEncoder(), reduction=4)
    inputs = torch.randn(2, 3, 32, 32)
    baseline = encoder.encoder(inputs)
    attended = encoder(inputs)
    assert encoder.out_channels == (3, 8, 16)
    assert encoder.output_stride == 4
    assert torch.equal(attended[0], baseline[0])
    assert torch.equal(attended[1], baseline[1])
    assert attended[-1].shape == baseline[-1].shape
    assert not torch.equal(attended[-1], baseline[-1])
