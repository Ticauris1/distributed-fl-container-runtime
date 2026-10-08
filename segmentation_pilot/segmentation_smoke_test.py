import torch
import segmentation_models_pytorch as smp

device = torch.device(
    "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,  # Keep raw logits for BCEWithLogitsLoss
).to(device)

dummy_input = torch.randn(1, 3, 256, 256, device=device)

model.eval()
with torch.no_grad():
    logits = model(dummy_input)
    probabilities = torch.sigmoid(logits)
    predicted_mask = probabilities >= 0.5

print("Device:", device)
print("Input shape:", dummy_input.shape)
print("Logit shape:", logits.shape)
print("Probability shape:", probabilities.shape)
print("Mask shape:", predicted_mask.shape)