"""JSON and pose loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import Pose


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_pose_map(path: str | Path) -> dict[str, Pose]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise TypeError("Pose JSON must be a list of pose records.")
    poses = [Pose.from_record(item) for item in payload]
    return {pose.instr_id: pose for pose in poses}


def dump_pose_records(poses: list[Pose], path: str | Path) -> None:
    dump_json([pose.to_record() for pose in poses], path)


def unwrap_data_list(payload: Any, *preferred_keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return list(value)
    raise TypeError("Expected a list payload or a dict containing a list field.")

