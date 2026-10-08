"""Named TabPy functions for advanced Tableau calculations."""

from math import sqrt
from statistics import mean, stdev


def mean_value(values):
    clean = [float(value) for value in values if value is not None]
    result = mean(clean) if clean else float("nan")
    return [result] * len(values)


def confidence_interval_95(values):
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        half_width = 0.0
    else:
        half_width = 1.96 * stdev(clean) / sqrt(len(clean))
    return [half_width] * len(values)


def paired_delta(left, right):
    return [
        float(right_value) - float(left_value)
        for left_value, right_value in zip(left, right)
    ]


def rolling_mean(values, window=5):
    clean = [float(value) for value in values]
    width = max(1, int(window))
    return [mean(clean[max(0, index - width + 1):index + 1]) for index in range(len(clean))]


def pareto_score(dice, milliseconds_per_image, dice_weight=0.8):
    accuracy = [float(value) for value in dice]
    latency = [float(value) for value in milliseconds_per_image]
    if not accuracy:
        return []
    latency_min, latency_max = min(latency), max(latency)
    span = latency_max - latency_min
    speed = [1.0 if span == 0 else 1.0 - (value - latency_min) / span for value in latency]
    weight = float(dice_weight)
    return [weight * acc + (1.0 - weight) * fast for acc, fast in zip(accuracy, speed)]
