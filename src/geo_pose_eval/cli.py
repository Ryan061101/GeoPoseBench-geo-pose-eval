"""Command line interface for geo-pose-eval."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .connectivity import attach_positions
from .encoders import build_reference_lookup, encode_prediction_poses, encode_reference_poses
from .evaluator import EvaluationConfig, evaluate_pose_files
from .io import dump_pose_records, load_json, unwrap_data_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified pose encoding and geometric localization evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference_parser = subparsers.add_parser("encode-reference", help="Normalize reference annotations into Pose records.")
    reference_parser.add_argument("input_json", type=Path)
    reference_parser.add_argument("output_json", type=Path)
    reference_parser.add_argument("--language", type=str, default=None)
    reference_parser.add_argument("--connectivity-dir", type=Path, default=None)

    prediction_parser = subparsers.add_parser("encode-prediction", help="Normalize prediction outputs into Pose records.")
    prediction_parser.add_argument("input_json", type=Path)
    prediction_parser.add_argument("output_json", type=Path)
    prediction_parser.add_argument("--reference-json", type=Path, default=None)
    prediction_parser.add_argument("--connectivity-dir", type=Path, default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run visibility-aware geometric evaluation.")
    evaluate_parser.add_argument("reference_pose_json", type=Path)
    evaluate_parser.add_argument("prediction_pose_json", type=Path)
    evaluate_parser.add_argument("output_json", type=Path)
    evaluate_parser.add_argument("--width", type=int, default=640)
    evaluate_parser.add_argument("--height", type=int, default=480)
    evaluate_parser.add_argument("--hfov", type=float, default=1.5707963267948966)
    evaluate_parser.add_argument("--vfov", type=float, default=1.0471975511965976)
    evaluate_parser.add_argument("--alpha-h", type=float, default=0.1)
    evaluate_parser.add_argument("--alpha-v", type=float, default=0.1)
    evaluate_parser.add_argument("--sigma", type=float, default=50.0)
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--landmark-distance", type=float, default=1.0)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "encode-reference":
        annotations = unwrap_data_list(load_json(args.input_json), "data", "annotations")
        poses = encode_reference_poses(annotations, language=args.language)
        if args.connectivity_dir is not None:
            poses = attach_positions(poses, args.connectivity_dir)
        dump_pose_records(poses, args.output_json)
        return 0

    if args.command == "encode-prediction":
        predictions = unwrap_data_list(load_json(args.input_json), "predictions", "data")
        reference_lookup = None
        if args.reference_json is not None:
            annotations = unwrap_data_list(load_json(args.reference_json), "data", "annotations")
            reference_lookup = build_reference_lookup(annotations)
        poses = encode_prediction_poses(predictions, reference_lookup=reference_lookup)
        if args.connectivity_dir is not None:
            poses = attach_positions(poses, args.connectivity_dir)
        dump_pose_records(poses, args.output_json)
        return 0

    config = EvaluationConfig(
        width=args.width,
        height=args.height,
        hfov=args.hfov,
        vfov=args.vfov,
        alpha_h=args.alpha_h,
        alpha_v=args.alpha_v,
        sigma=args.sigma,
        iou_threshold=args.iou_threshold,
        landmark_distance=args.landmark_distance,
    )
    evaluate_pose_files(
        str(args.reference_pose_json),
        str(args.prediction_pose_json),
        str(args.output_json),
        config=config,
    )
    return 0

