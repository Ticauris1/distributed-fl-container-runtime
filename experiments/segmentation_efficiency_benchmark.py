"""Benchmark saved segmentation checkpoints without retraining.

The benchmark excludes image loading and preprocessing from timed inference by
preloading the CCSS test tensors. Device work is synchronized around every
measurement so MPS/CUDA's asynchronous execution does not under-report time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.segmentation import SegmentationDataset
from models.segmentation import make_segmentation_model


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "segmentation" / "efficiency_benchmark"
)
DEFAULT_CHECKPOINTS = {
    "efficientnet_b3": PROJECT_ROOT / "results" / "segmentation"
    / "structured_comparison" / "ccss_unetplusplus_efficientnet_b0_b3_comparison"
    / "unetplusplus_efficientnet_b3" / "seed_42" / "unetplusplus_efficientnet-b3"
    / "checkpoints" / "best_unetplusplus.pth",
    "inceptionresnetv2": PROJECT_ROOT / "results" / "segmentation"
    / "structured_comparison" / "ccss_inceptionresnetv2_decoder_comparison"
    / "ccss_unetplusplus_inceptionresnetv2" / "seed_42"
    / "unetplusplus_inceptionresnetv2" / "checkpoints" / "best_unetplusplus.pth",
}


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def current_allocated_bytes(device):
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    return None


def clear_device_memory(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def preload_test_batches(batch_size):
    dataset = SegmentationDataset(
        config.CCSS_SEGMENTATION_MANIFEST,
        "test",
        config.SEGMENTATION_IMAGE_SIZE,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return [images.contiguous() for images, _ in loader]


def percentile(values, proportion):
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * proportion)))
    return ordered[position]


def benchmark_checkpoint(
    label, encoder, checkpoint, batches, device, warmup_iterations, timed_iterations
):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    clear_device_memory(device)
    model = make_segmentation_model(
        "unetplusplus", device, encoder, encoder_weights=None
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    synchronize(device)
    memory_after_model_load = current_allocated_bytes(device)

    with torch.inference_mode():
        for index in range(int(warmup_iterations)):
            images = batches[index % len(batches)].to(device)
            _ = model(images)
            synchronize(device)

        timings = []
        image_count = 0
        observed_allocations = []
        for index in range(int(timed_iterations)):
            images = batches[index % len(batches)].to(device)
            synchronize(device)
            started = time.perf_counter()
            _ = model(images)
            synchronize(device)
            timings.append(time.perf_counter() - started)
            image_count += int(images.shape[0])
            allocation = current_allocated_bytes(device)
            if allocation is not None:
                observed_allocations.append(allocation)

    elapsed = sum(timings)
    observed_peak = max(observed_allocations) if observed_allocations else None
    observed_increment = (
        observed_peak - memory_after_model_load
        if observed_peak is not None and memory_after_model_load is not None
        else None
    )
    result = {
        "label": label,
        "model": "unetplusplus",
        "encoder": encoder,
        "device": str(device),
        "image_size": int(config.SEGMENTATION_IMAGE_SIZE),
        "batch_size": int(batches[0].shape[0]),
        "warmup_iterations": int(warmup_iterations),
        "timed_iterations": int(timed_iterations),
        "timed_images": image_count,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_megabytes": checkpoint.stat().st_size / (1024 ** 2),
        "total_timed_seconds": elapsed,
        "mean_batch_milliseconds": statistics.mean(timings) * 1000.0,
        "median_batch_milliseconds": statistics.median(timings) * 1000.0,
        "p95_batch_milliseconds": percentile(timings, 0.95) * 1000.0,
        "mean_inference_milliseconds_per_image": elapsed * 1000.0 / image_count,
        "images_per_second": image_count / elapsed,
        # PyTorch MPS has no resettable peak-memory statistic. This is the
        # largest current allocation sampled immediately after a forward pass.
        "memory_measurement": (
            "observed_current_allocated_after_forward"
            if observed_peak is not None else "unavailable"
        ),
        "model_loaded_allocated_bytes": memory_after_model_load,
        "observed_peak_allocated_bytes": observed_peak,
        "observed_forward_increment_bytes": observed_increment,
        "checkpoint_path": str(checkpoint),
    }
    del model
    clear_device_memory(device)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--timed-iterations", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    config.set_seed(42)
    device = config.DEVICE
    batches = preload_test_batches(args.batch_size)
    conditions = [
        ("efficientnet_b3", "efficientnet_b3", DEFAULT_CHECKPOINTS["efficientnet_b3"]),
        (
            "inceptionresnetv2",
            "inceptionresnetv2",
            DEFAULT_CHECKPOINTS["inceptionresnetv2"],
        ),
    ]
    results = []
    for label, encoder, checkpoint in conditions:
        print(f"[efficiency] benchmarking {label} on {device}")
        result = benchmark_checkpoint(
            label,
            encoder,
            checkpoint,
            batches,
            device,
            args.warmup_iterations,
            args.timed_iterations,
        )
        results.append(result)
        print(
            f"[efficiency] {label}: "
            f"{result['mean_inference_milliseconds_per_image']:.2f} ms/image, "
            f"{result['images_per_second']:.2f} images/s"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "checkpoint_efficiency_summary.csv"
    json_path = args.output_dir / "checkpoint_efficiency_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(f"[efficiency] CSV: {csv_path}")
    print(f"[efficiency] JSON: {json_path}")


if __name__ == "__main__":
    main()
