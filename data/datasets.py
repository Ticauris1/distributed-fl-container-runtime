"""PyTorch dataset classes."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path, label = self.image_paths[index], self.labels[index]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as error:
            print(f"Warning: Could not load image {path}. Skipping. Error: {error}")
            return torch.zeros((3, 224, 224)), label
        return image, label


def coerce_features_to_float2d(features):
    array = np.asarray(features, dtype=object)
    if array.dtype == object:
        try:
            array = np.stack(array, axis=0)
        except Exception:
            array = np.array(
                [np.asarray(item).ravel() for item in array], dtype=np.float32
            )
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        features = coerce_features_to_float2d(features)
        labels = np.asarray(labels, dtype=np.int64)
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
