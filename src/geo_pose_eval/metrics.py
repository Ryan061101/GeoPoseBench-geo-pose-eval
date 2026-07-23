"""Metric helpers for geometric localization."""

from __future__ import annotations

from math import exp, sqrt

from .schemas import BoundingBox


def intersection_over_union(lhs: BoundingBox, rhs: BoundingBox) -> float:
    inter_xmin = max(lhs.xmin, rhs.xmin)
    inter_ymin = max(lhs.ymin, rhs.ymin)
    inter_xmax = min(lhs.xmax, rhs.xmax)
    inter_ymax = min(lhs.ymax, rhs.ymax)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    intersection = inter_w * inter_h
    union = lhs.area + rhs.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def center_error(lhs: BoundingBox, rhs: BoundingBox) -> float:
    u1, v1 = lhs.center
    u2, v2 = rhs.center
    return sqrt((u1 - u2) ** 2 + (v1 - v2) ** 2)


def geometry_success(iou: float, threshold: float = 0.5) -> float:
    return 1.0 if iou >= threshold else 0.0


def geo_rgs(iou: float, visibility: float, center_err: float, sigma: float) -> float:
    return iou * visibility * exp(-center_err / sigma)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)

