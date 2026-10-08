"""Paper-inspired lightweight CNN construction and checkpoint handling."""

import os
from pathlib import Path
import re
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore


class LightweightCNN(nn.Module):
    """Reference lightweight model with the original ~0.69M parameter target."""

    def __init__(self, num_classes):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 212), nn.ReLU(), nn.Dropout(0.7), nn.Linear(212, num_classes)
        )

    def forward(self, inputs):
        output = self.conv2(self.conv1(inputs))
        output = self.conv3(output)
        return self.classifier(output)


def verify_model_complexity(model):
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    params_in_millions = total_params / 1e6
    print("\n" + "=" * 30)
    print("📊 MODEL COMPLEXITY REPORT")
    print("=" * 30)
    print(f"Total Trainable Params: {total_params:,}")
    print(f"Total Params (Millions): {params_in_millions:.3f}M")
    if 0.60 <= params_in_millions <= 0.80:
        print("✅ Status: Matches Paper Target (~0.69M)")
    else:
        print("⚠️ Status: Deviation from Paper Target")
    print("=" * 30 + "\n")
    return total_params


def train_and_save_lightweight(
    train_loader, val_loader, num_classes, device, save_path,
    model_name="paper_lightweight", epochs=100,
):
    del model_name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_train_images = len(train_loader.dataset)
    num_val_images = len(val_loader.dataset)
    model = LightweightCNN(num_classes).to(device)
    verify_model_complexity(model)
    print(f"📊 Training Data: {num_train_images} images")
    print(f"📊 Validation Data: {num_val_images} images")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5, factor=0.5)
    best_accuracy = 0.0
    print(f"--- 🚀 Training Paper-Lightweight Model (Target: {save_path}) ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / num_train_images
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        val_accuracy = 100 * correct / total
        scheduler.step(val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch + 1:02d}: Loss: {epoch_loss:.4f} | New Best Accuracy! {val_accuracy:.2f}% - Saved")
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1:03d}/{epochs}] Loss: {epoch_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
    print(f"✅ Training Complete. Best Accuracy Saved: {best_accuracy:.2f}%")
    return model


def _dataset_checkpoint_path(checkpoint_root, dataset_name):
    if not dataset_name:
        raise ValueError("dataset_name is required to select a dataset-specific checkpoint")
    dataset_dir = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    if not dataset_dir:
        raise ValueError(f"Invalid dataset name: {dataset_name!r}")
    return Path(checkpoint_root) / dataset_dir / "paper_lightweight_best.pth"


def _checkpoint_matches_model(state_dict, model):
    model_state = model.state_dict()
    return set(state_dict) == set(model_state) and all(
        model_state[key].shape == value.shape for key, value in state_dict.items()
    )


def get_ready_lightweight_model(
    num_classes,
    device,
    train_loader=None,
    val_loader=None,
    checkpoint_root=None,
    dataset_name=None,
    epochs=100,
):
    model = LightweightCNN(num_classes).to(device)
    checkpoint_root = checkpoint_root or Path("saved_models") / "paper_lightweight"
    checkpoint_path = _dataset_checkpoint_path(checkpoint_root, dataset_name)
    print(f"[Lightweight checkpoint] Dataset '{dataset_name}' ({num_classes} classes): {checkpoint_path}")
    if checkpoint_path.exists():
        print(f"✅ Loading verified lightweight baseline from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if _checkpoint_matches_model(state_dict, model):
            model.load_state_dict(state_dict)
            model.eval()
            return model
        print(
            f"⚠️ Checkpoint at {checkpoint_path} does not match the {num_classes}-class "
            "model. A new dataset-specific checkpoint will be trained."
        )
    if train_loader is None or val_loader is None:
        raise ValueError(
            f"❌ No compatible LW checkpoint found for dataset '{dataset_name}'. "
            "Both a dedicated training loader and validation loader are required."
        )
    print("⚠️ Warning: Pre-trained baseline not found. Training now...")
    model = train_and_save_lightweight(
        train_loader,
        val_loader,
        num_classes,
        device,
        str(checkpoint_path),
        epochs=epochs,
    )
    model.eval()
    return model


def load_lightweight_model_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model_state = model.state_dict()
    filtered = {}
    for key, value in state.items():
        if key in ("fc5.weight", "fc5.bias") and value.shape != model_state[key].shape:
            print(f"⚠️ Skipping {key}: Checkpoint shape {value.shape} != Model shape {model_state[key].shape}")
            continue
        filtered[key] = value
    model.load_state_dict(filtered, strict=False)
    print(f"✅ Successfully loaded checkpoint from {path} (partial load enabled)")
    return model
