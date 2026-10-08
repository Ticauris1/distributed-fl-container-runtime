"""Local client training primitives."""

from typing import Dict, List
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm.auto import tqdm

from models.classifiers import ImageClassifier


def epoch_train(
    model, loader, optimizer, criterion, device, use_amp=True, max_grad_norm=1.0,
):
    model.train()
    is_cuda = device.type == "cuda"
    amp_enabled = use_amp and is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if is_cuda else None
    running_loss = running_correct = total = 0.0
    progress = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in progress:
        inputs = inputs.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                if not torch.isfinite(logits).all():
                    invalid = int((~torch.isfinite(logits)).sum().item())
                    raise FloatingPointError(
                        f"Local training produced {invalid} non-finite logits."
                    )
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Local training produced a non-finite loss: {loss.detach().item()}."
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm, error_if_nonfinite=True,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                invalid = int((~torch.isfinite(logits)).sum().item())
                raise FloatingPointError(
                    f"Local training produced {invalid} non-finite logits."
                )
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Local training produced a non-finite loss: {loss.detach().item()}."
                )
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm, error_if_nonfinite=True,
                )
            optimizer.step()
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(1) == labels).sum().item()
        total += batch_size
        progress.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{running_correct / max(total, 1):.4f}")
    return running_loss / max(total, 1), running_correct / max(total, 1)


class GradualUnfreezer:
    def __init__(self, model: ImageClassifier, unfreeze_schedule: Dict[int, int]):
        self.model = model
        self.schedule = unfreeze_schedule
        self.backbone_layers: List[nn.Module] = list(model.backbone.children())
        self.current_block_to_unfreeze = len(self.backbone_layers) - 1
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad = False

    def step(self, epoch: int, optimizer: Optimizer, base_learning_rate: float):
        if epoch not in self.schedule:
            return
        blocks_to_unfreeze = self.schedule[epoch]
        print(f"\n🔥 Epoch {epoch}: Unfreezing {blocks_to_unfreeze} layer block(s)...")
        for _ in range(blocks_to_unfreeze):
            if self.current_block_to_unfreeze < 0:
                print("  - All blocks already unfrozen.")
                break
            block = self.backbone_layers[self.current_block_to_unfreeze]
            for parameter in block.parameters():
                parameter.requires_grad = True
            optimizer.add_param_group({"params": list(block.parameters()), "lr": base_learning_rate / 10})
            print(f"  - Unfroze block {self.current_block_to_unfreeze} and added to optimizer.")
            self.current_block_to_unfreeze -= 1
