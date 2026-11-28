"""P1.2: low-complexity illumination evidence, never a geometry substitute.

Fit one gain/offset on training pixels; test on disjoint pixels and tiles.
Unreliable correspondence, poor texture or poor fit always abstains. Brightness
evidence may downgrade edit claims; it never promotes unknown to motion/edit.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class BrightnessParameters:
    min_gain: float = .5
    max_gain: float = 2.
    max_abs_offset: float = .3
    min_texture_std: float = .035
    min_fit_pixels: int = 1000
    max_fit_pixels: int = 20000
    huber_delta: float = .02
    fit_iterations: int = 5
    min_gradient_correlation: float = .9
    max_validation_mad: float = .012
    max_validation_residual_ratio: float = .35
    min_brightness_effect: float = .002
    min_tile_support_fraction: float = .8
    min_group_support_fraction: float = .8

    def validate(self):
        d = asdict(self)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in d.values()):
            raise ValueError("Brightness parameters must be positive finite numbers")
        if self.min_gain >= self.max_gain:
            raise ValueError("min_gain must be less than max_gain")
        for key in ("min_gradient_correlation", "max_validation_residual_ratio", "min_tile_support_fraction", "min_group_support_fraction"):
            if d[key] > 1:
                raise ValueError(key + " must be <=1")
        for key in ("min_fit_pixels", "max_fit_pixels", "fit_iterations"):
            if not isinstance(d[key], int):
                raise ValueError(key + " must be an integer")


def _correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.


def brightness_metrics(a, b, geometry, base_params, p=BrightnessParameters()):
    """Evaluate b ~= clip(gain * warp(a) + offset, 0, 1) on valid overlap.

    Units: gain dimensionless, offset and residual normalized 0..1 grayscale.
    No fit across cuts/low-registration pairs; no histogram-only explanations.
    """
    p.validate()
    out = {"supported": False, "reason": "unreliable_correspondence", "gain": None,
           "offset": None, "validation_mad": None, "validation_residual_ratio": None,
           "gradient_correlation": None, "brightness_effect": None,
           "tile_support_fraction": None, "train_pixels": 0, "validation_pixels": 0}
    if not geometry.get("registration_valid"):
        return out
    if geometry.get("phase_response", 0) < base_params.min_phase_response or geometry.get("overlap", 0) < base_params.min_overlap:
        return out
    h, w = a.shape
    m = np.float32([[1, 0, geometry["dx"]], [0, 1, geometry["dy"]]])
    x = cv2.warpAffine(a.astype(np.float32) / 255., m, (w, h))
    y = b.astype(np.float32) / 255.
    support = cv2.warpAffine(np.ones_like(x), m, (w, h)) > .999
    # Gradients need a full 3x3 supported neighborhood, not padded borders.
    support = cv2.erode(support.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    yy, xx = np.indices(a.shape)
    train = support & (((xx + yy) % 2) == 0) & (x > .02) & (x < .98) & (y > .02) & (y < .98)
    valid = support & (((xx + yy) % 2) == 1)
    out.update(train_pixels=int(train.sum()), validation_pixels=int(valid.sum()))
    if train.sum() < p.min_fit_pixels or valid.sum() < p.min_fit_pixels or float(x[train].std()) < p.min_texture_std:
        out["reason"] = "insufficient_texture_or_pixels"
        return out
    tx, ty = x[train].astype(np.float64), y[train].astype(np.float64)
    stride = max(1, math.ceil(len(tx) / p.max_fit_pixels))
    tx, ty = tx[::stride], ty[::stride]
    design = np.column_stack((tx, np.ones_like(tx)))
    coef = np.linalg.lstsq(design, ty, rcond=None)[0]
    for _ in range(p.fit_iterations):
        err = ty - design @ coef
        weights = np.minimum(1., p.huber_delta / np.maximum(np.abs(err), 1e-9))
        sq = np.sqrt(weights)
        coef = np.linalg.lstsq(design * sq[:, None], ty * sq, rcond=None)[0]
    gain, offset = map(float, coef)
    out.update(gain=gain, offset=offset)
    if not p.min_gain <= gain <= p.max_gain or abs(offset) > p.max_abs_offset:
        out["reason"] = "gain_or_offset_out_of_range"
        return out
    predicted = np.clip(gain * x + offset, 0, 1)
    before = np.abs(x - y)
    after = np.abs(predicted - y)
    residual = float(after[valid].mean())
    ratio = residual / max(float(before[valid].mean()), 1e-6)
    effect = float(np.abs(predicted[valid] - x[valid]).mean())
    gx = np.concatenate([cv2.Sobel(x, cv2.CV_32F, 1, 0)[support], cv2.Sobel(x, cv2.CV_32F, 0, 1)[support]])
    gy = np.concatenate([cv2.Sobel(y, cv2.CV_32F, 1, 0)[support], cv2.Sobel(y, cv2.CV_32F, 0, 1)[support]])
    corr = _correlation(gx, gy)
    tiles = []
    for rows in np.array_split(np.arange(h), 4):
        for cols in np.array_split(np.arange(w), 6):
            sl = np.ix_(rows, cols)
            mask = valid[sl]
            if mask.sum() < 50:
                continue
            r0, r1 = float(before[sl][mask].mean()), float(after[sl][mask].mean())
            # Unchanged tiles are neutral; changed tiles must independently fit.
            if r0 >= base_params.activity_mad:
                tiles.append(r1 <= p.max_validation_mad and r1 / max(r0, 1e-6) <= p.max_validation_residual_ratio)
    fraction = float(np.mean(tiles)) if tiles else 0.
    out.update(validation_mad=residual, validation_residual_ratio=ratio,
               brightness_effect=effect, gradient_correlation=corr,
               tile_support_fraction=fraction, evaluated_changed_tiles=len(tiles))
    if corr < p.min_gradient_correlation:
        out["reason"] = "structure_inconsistent"
    elif effect < p.min_brightness_effect:
        out["reason"] = "no_meaningful_brightness_change"
    elif residual > p.max_validation_mad or ratio > p.max_validation_residual_ratio or fraction < p.min_tile_support_fraction:
        out["reason"] = "brightness_model_leaves_spatial_residual"
    else:
        out.update(supported=True, reason="gain_offset_explains_validated_overlap")
    return out
