"""Matterport connectivity helpers for pose coordinates."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import Pose


def position_from_pose(pose: list[float]) -> tuple[float, float, float]:
    if len(pose) < 12:
        raise ValueError("Connectivity pose must contain at least 12 values.")
    return (float(pose[3]), float(pose[7]), float(pose[11]))


@lru_cache(maxsize=None)
def load_scan_positions(connectivity_dir: str, scan: str) -> dict[str, tuple[float, float, float]]:
    path = Path(connectivity_dir) / f"{scan}_connectivity.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    positions: dict[str, tuple[float, float, float]] = {}
    for item in data:
        if not item.get("included", False):
            continue
        positions[str(item["image_id"])] = position_from_pose(item["pose"])
    return positions


def attach_positions(
    poses: list[Pose],
    connectivity_dir: str | Path,
    *,
    strict: bool = True,
) -> list[Pose]:
    root = str(Path(connectivity_dir))
    enriched: list[Pose] = []
    for pose in poses:
        positions = load_scan_positions(root, pose.scan)
        xyz = positions.get(pose.viewpoint)
        if xyz is None:
            if strict:
                raise KeyError(f"Missing connectivity position for {pose.scan}/{pose.viewpoint}.")
            enriched.append(pose)
            continue
        enriched.append(
            Pose(
                instr_id=pose.instr_id,
                scan=pose.scan,
                viewpoint=pose.viewpoint,
                heading=pose.heading,
                elevation=pose.elevation,
                x=xyz[0],
                y=xyz[1],
                z=xyz[2],
                path_id=pose.path_id,
                language=pose.language,
                instruction=pose.instruction,
            )
        )
    return enriched


def attach_positions_to_records(
    records: list[dict[str, Any]],
    connectivity_dir: str | Path,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    poses = [Pose.from_record(record) for record in records]
    return [pose.to_record() for pose in attach_positions(poses, connectivity_dir, strict=strict)]

