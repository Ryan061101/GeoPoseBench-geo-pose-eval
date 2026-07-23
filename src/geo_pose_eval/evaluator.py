"""Visibility-aware geometric localization evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .io import dump_json, load_pose_map
from .metrics import center_error, geo_rgs, geometry_success, intersection_over_union, mean_or_none
from .projection import build_intrinsics, landmark_center_from_reference, project_landmark
from .schemas import Pose, ProjectionResult


@dataclass(slots=True)
class EvaluationConfig:
    width: int = 640
    height: int = 480
    hfov: float = 1.5707963267948966
    vfov: float = 1.0471975511965976
    alpha_h: float = 0.1
    alpha_v: float = 0.1
    sigma: float = 50.0
    iou_threshold: float = 0.5
    landmark_distance: float = 1.0


def evaluate_pose_maps(
    reference_pose_map: Mapping[str, Pose],
    prediction_pose_map: Mapping[str, Pose],
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    active_config = config or EvaluationConfig()
    intrinsics = build_intrinsics(
        width=active_config.width,
        height=active_config.height,
        hfov=active_config.hfov,
        vfov=active_config.vfov,
    )

    samples: list[dict[str, Any]] = []
    visible_ious: list[float] = []
    visible_successes: list[float] = []
    visible_rgs: list[float] = []
    overall_rgs_scores: list[float] = []
    visible_count = 0
    missing_predictions = 0
    matched_predictions = 0

    for instr_id, reference_pose in reference_pose_map.items():
        prediction_pose = prediction_pose_map.get(instr_id)
        sample: dict[str, Any] = {
            "instr_id": instr_id,
            "scan": reference_pose.scan,
        }

        if prediction_pose is None:
            missing_predictions += 1
            sample.update(
                {
                    "status": "missing_prediction",
                    "visible": False,
                    "gt_bbox": None,
                    "pred_bbox": None,
                    "reference_visibility": 0.0,
                    "prediction_visibility": 0.0,
                    "iou": None,
                    "geometry_success": 0.0,
                    "center_error": None,
                    "geo_rgs": 0.0,
                }
            )
            samples.append(sample)
            overall_rgs_scores.append(0.0)
            continue

        matched_predictions += 1
        landmark_world = landmark_center_from_reference(reference_pose, distance=active_config.landmark_distance)
        gt_projection, gt_reason = _project_safe(
            landmark_world,
            reference_pose,
            intrinsics,
            active_config.alpha_h,
            active_config.alpha_v,
        )
        pred_projection, pred_reason = _project_safe(
            landmark_world,
            prediction_pose,
            intrinsics,
            active_config.alpha_h,
            active_config.alpha_v,
        )

        visible = _is_visible(gt_projection) and _is_visible(pred_projection)
        if visible:
            visible_count += 1
            iou = intersection_over_union(gt_projection.bbox, pred_projection.bbox)
            success = geometry_success(iou, active_config.iou_threshold)
            err = center_error(gt_projection.bbox, pred_projection.bbox)
            score = geo_rgs(iou=iou, visibility=pred_projection.visible_ratio, center_err=err, sigma=active_config.sigma)
            status = "visible"
            visible_ious.append(iou)
            visible_successes.append(success)
            visible_rgs.append(score)
        else:
            iou = None
            success = 0.0
            err = None
            score = 0.0
            status = _resolve_invisible_status(gt_projection, pred_projection, gt_reason, pred_reason)

        sample.update(
            {
                "status": status,
                "visible": visible,
                "gt_bbox": gt_projection.bbox.to_record() if gt_projection is not None else None,
                "pred_bbox": pred_projection.bbox.to_record() if pred_projection is not None else None,
                "reference_visibility": gt_projection.visible_ratio if gt_projection is not None else 0.0,
                "prediction_visibility": pred_projection.visible_ratio if pred_projection is not None else 0.0,
                "iou": iou,
                "geometry_success": success,
                "center_error": err,
                "geo_rgs": score,
            }
        )
        samples.append(sample)
        overall_rgs_scores.append(score)

    total_samples = len(reference_pose_map)
    summary = {
        "total_samples": total_samples,
        "matched_predictions": matched_predictions,
        "missing_predictions": missing_predictions,
        "visible_samples": visible_count,
        "invisible_samples": total_samples - visible_count,
        "visibility_rate": (visible_count / total_samples) if total_samples else 0.0,
        "mean_iou_visible": mean_or_none(visible_ious),
        "geometry_success_rate_visible": mean_or_none(visible_successes),
        "mean_rgs_visible": mean_or_none(visible_rgs),
        "overall_rgs": (sum(overall_rgs_scores) / total_samples) if total_samples else 0.0,
        "iou_threshold": active_config.iou_threshold,
    }
    return {"summary": summary, "samples": samples}


def evaluate_pose_files(
    reference_pose_json: str,
    prediction_pose_json: str,
    output_json: str,
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    report = evaluate_pose_maps(
        load_pose_map(reference_pose_json),
        load_pose_map(prediction_pose_json),
        config=config,
    )
    dump_json(report, output_json)
    return report


def _project_safe(
    landmark_world: tuple[float, float, float],
    pose: Pose,
    intrinsics,
    alpha_h: float,
    alpha_v: float,
) -> tuple[ProjectionResult | None, str | None]:
    try:
        return project_landmark(landmark_world, pose, intrinsics, alpha_h, alpha_v), None
    except ValueError:
        return None, "not_projectable"


def _is_visible(projection: ProjectionResult | None) -> bool:
    return projection is not None and projection.visible_ratio > 0.0


def _resolve_invisible_status(
    gt_projection: ProjectionResult | None,
    pred_projection: ProjectionResult | None,
    gt_reason: str | None,
    pred_reason: str | None,
) -> str:
    if gt_projection is None:
        return "reference_not_projectable" if gt_reason == "not_projectable" else "reference_invisible"
    if pred_projection is None:
        return "prediction_not_projectable" if pred_reason == "not_projectable" else "prediction_invisible"
    if gt_projection.visible_ratio <= 0.0:
        return "reference_outside_frame"
    return "prediction_outside_frame"

