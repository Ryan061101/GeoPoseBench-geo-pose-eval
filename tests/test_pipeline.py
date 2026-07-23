"""Unit tests for the standalone geo-pose-eval package."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geo_pose_eval.connectivity import attach_positions, position_from_pose
from geo_pose_eval.encoders import build_reference_lookup, encode_prediction_poses, encode_reference_poses
from geo_pose_eval.evaluator import EvaluationConfig, evaluate_pose_maps
from geo_pose_eval.metrics import geometry_success, intersection_over_union
from geo_pose_eval.projection import build_bbox, build_intrinsics, clip_bbox, visibility_ratio
from geo_pose_eval.schemas import BoundingBox, Pose


class GeoPoseEvalTests(unittest.TestCase):
    def test_reference_encoding_picks_final_pose(self) -> None:
        payload = [
            {
                "instr_id": "123",
                "scan": "scanA",
                "sub_paths": [["vp0", "vp1"], ["vp2", "vp3"]],
                "heading": 0.25,
                "headings": [0.25, 0.5],
                "elevation": -0.1,
                "language": "en-US",
                "instruction": "Walk to the lamp",
            }
        ]
        result = encode_reference_poses(payload)
        self.assertEqual(result[0].viewpoint, "vp3")
        self.assertEqual(result[0].heading, 0.5)
        self.assertEqual(result[0].language, "en-US")

    def test_prediction_encoding_recovers_instruction_id(self) -> None:
        predictions = [
            {
                "path_id": 7,
                "instructions": ['"Walk to the lamp"'],
                "trajectory": [
                    {"scan": "scanA", "viewpoint": "vp0", "heading": 0.1, "elevation": 0.0},
                    {"scan": "scanA", "viewpoint": "vp1", "heading": 0.2, "elevation": 0.3},
                ],
            }
        ]
        annotations = [
            {
                "path_id": 7,
                "instruction_id": "123",
                "language": "en-US",
                "instruction": "Walk to the lamp",
            }
        ]
        result = encode_prediction_poses(predictions, reference_lookup=build_reference_lookup(annotations))
        self.assertEqual(result[0].viewpoint, "vp1")
        self.assertAlmostEqual(result[0].elevation, 0.3)
        self.assertEqual(result[0].instr_id, "123")

    def test_prediction_encoding_supports_duet_pose_history(self) -> None:
        predictions = [
            {
                "instr_id": "5676_0",
                "poses": [
                    {
                        "scan": "scanA",
                        "viewpoint": "vp0",
                        "heading": 0.5,
                        "elevation": 0.1,
                    },
                    {
                        "scan": "scanA",
                        "viewpoint": "vp1",
                        "heading": 1.0,
                        "elevation": 0.0,
                    },
                ],
                "trajectory": [["vp0"], ["vp1"]],
            }
        ]
        result = encode_prediction_poses(predictions)
        self.assertEqual(result[0].instr_id, "5676_0")
        self.assertEqual(result[0].scan, "scanA")
        self.assertEqual(result[0].viewpoint, "vp1")
        self.assertAlmostEqual(result[0].heading, 1.0)

    def test_attach_positions_uses_connectivity_coordinates(self) -> None:
        poses = [Pose(instr_id="1", scan="scanA", viewpoint="vp1", heading=0.0, elevation=0.0)]
        with patch("geo_pose_eval.connectivity.load_scan_positions", return_value={"vp1": (1.0, 2.0, 3.0)}):
            enriched = attach_positions(poses, "connectivity")
        self.assertEqual(enriched[0].position, (1.0, 2.0, 3.0))

    def test_evaluator_splits_visible_and_invisible_samples(self) -> None:
        reference_pose_map = {
            "visible": Pose("visible", "scanA", "vp1", 0.0, 0.0, x=0.0, y=0.0, z=0.0),
            "hidden": Pose("hidden", "scanA", "vp1", 0.0, 0.0, x=0.0, y=0.0, z=0.0),
        }
        prediction_pose_map = {
            "visible": Pose("visible", "scanA", "vp1", 0.0, 0.0, x=0.0, y=0.0, z=0.0),
            "hidden": Pose("hidden", "scanA", "vp1", math.pi, 0.0, x=0.0, y=0.0, z=0.0),
        }
        report = evaluate_pose_maps(reference_pose_map, prediction_pose_map, config=EvaluationConfig())
        self.assertEqual(report["summary"]["visible_samples"], 1)
        self.assertEqual(report["summary"]["invisible_samples"], 1)
        self.assertAlmostEqual(report["summary"]["mean_iou_visible"], 1.0)
        self.assertAlmostEqual(report["summary"]["geometry_success_rate_visible"], 1.0)
        self.assertAlmostEqual(report["summary"]["overall_rgs"], 0.5)
        self.assertEqual(report["samples"][0]["status"], "visible")
        self.assertEqual(report["samples"][1]["status"], "prediction_not_projectable")

    def test_bbox_visibility_and_success_metrics(self) -> None:
        intrinsics = build_intrinsics(640, 480, math.pi / 2.0, math.pi / 3.0)
        bbox = build_bbox(intrinsics.cx, intrinsics.cy, 0.1, 0.05, intrinsics)
        clipped = clip_bbox(BoundingBox(-10.0, 20.0, 110.0, 70.0), 100, 80)
        self.assertAlmostEqual(bbox.width, 2.0 * intrinsics.fx * math.tan(0.1))
        self.assertEqual(clipped.xmin, 0.0)
        self.assertEqual(clipped.xmax, 100.0)
        lhs = BoundingBox(0.0, 0.0, 100.0, 100.0)
        rhs = BoundingBox(0.0, 0.0, 100.0, 100.0)
        self.assertAlmostEqual(intersection_over_union(lhs, rhs), 1.0)
        self.assertEqual(geometry_success(0.6, threshold=0.5), 1.0)
        self.assertGreater(visibility_ratio(BoundingBox(-10.0, 0.0, 10.0, 20.0), 100, 100), 0.0)

    def test_position_from_pose_extracts_translation(self) -> None:
        pose = [1, 0, 0, 10.0, 0, 1, 0, 20.0, 0, 0, 1, 30.0, 0, 0, 0, 1]
        self.assertEqual(position_from_pose(pose), (10.0, 20.0, 30.0))


if __name__ == "__main__":
    unittest.main()
