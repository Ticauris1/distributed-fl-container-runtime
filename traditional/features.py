"""HOG and color-histogram feature transformers."""

import numpy as np
from skimage.feature import hog
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


def infer_square_side(dimension):
    side = int(round((dimension / 3.0) ** 0.5))
    if 3 * side * side != dimension:
        side = int(np.sqrt(dimension // 3))
    return max(side, 1)


def hogify_flat(features):
    features = np.asarray(features, dtype=np.float32)
    sample_count, dimension = features.shape
    side = infer_square_side(dimension)
    images = features.reshape(sample_count, 3, side, side).mean(axis=1)
    return np.asarray([
        hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
        for image in images
    ], dtype=np.float32)


def color_hist_flat(features, bins=32):
    features = np.asarray(features, dtype=np.float32)
    sample_count, dimension = features.shape
    side = infer_square_side(dimension)
    images = features.reshape(sample_count, 3, side, side)
    histograms = []
    for image in images:
        channels = [np.histogram(image[channel].ravel(), bins=bins, range=(0, 1), density=True)[0] for channel in range(3)]
        histograms.append(np.concatenate(channels))
    return np.asarray(histograms, dtype=np.float32)


def create_feature_union(bins=32):
    hog_transform = ("hog", FunctionTransformer(hogify_flat, validate=False))
    color_transform = ("color", FunctionTransformer(color_hist_flat, kw_args={"bins": bins}, validate=False))
    return FeatureUnion([hog_transform, color_transform])


HOG = ("hog", FunctionTransformer(hogify_flat, validate=False))
COLOR = ("color", FunctionTransformer(color_hist_flat, kw_args={"bins": 32}, validate=False))
FEATS = ("feats", FeatureUnion([HOG, COLOR]))
_infer_square_side = infer_square_side
