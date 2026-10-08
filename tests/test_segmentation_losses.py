import torch

from training.segmentation import make_segmentation_loss


def test_new_segmentation_losses_are_finite_and_differentiable():
    targets = (torch.rand(2, 1, 32, 32) > 0.85).float()
    for name in (
        "bce_dice_boundary", "focal_tversky", "weighted_bce_dice"
    ):
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        loss = make_segmentation_loss(name)(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def test_boundary_loss_penalizes_shifted_mask_more_than_matching_mask():
    criterion = make_segmentation_loss("bce_dice_boundary")
    targets = torch.zeros(1, 1, 16, 16)
    targets[:, :, 4:12, 4:12] = 1.0
    matching_logits = torch.where(targets.bool(), 8.0, -8.0)
    shifted = torch.zeros_like(targets)
    shifted[:, :, 5:13, 5:13] = 1.0
    shifted_logits = torch.where(shifted.bool(), 8.0, -8.0)
    assert criterion(matching_logits, targets) < criterion(shifted_logits, targets)


def test_weighted_bce_dice_increases_positive_pixel_penalty():
    targets = torch.zeros(1, 1, 8, 8)
    targets[:, :, 3, 3] = 1.0
    logits = torch.zeros_like(targets)
    ordinary = make_segmentation_loss("bce_dice")(logits, targets)
    weighted = make_segmentation_loss(
        "weighted_bce_dice", positive_class_weight=5.0
    )(logits, targets)
    assert weighted > ordinary
