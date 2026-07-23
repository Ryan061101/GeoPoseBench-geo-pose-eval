"""Pose normalization for reference and prediction inputs."""

from __future__ import annotations

from typing import Any

from .schemas import Pose


def normalize_instruction_text(text: str) -> str:
    collapsed = " ".join(text.strip().split())
    if len(collapsed) >= 2 and collapsed[0] == collapsed[-1] and collapsed[0] in {'"', "'"}:
        collapsed = collapsed[1:-1].strip()
    return collapsed


def build_reference_lookup(annotations: list[dict[str, Any]]) -> dict[tuple[Any, str], dict[str, Any]]:
    lookup: dict[tuple[Any, str], dict[str, Any]] = {}
    for item in annotations:
        instruction = item.get("instruction")
        path_id = item.get("path_id")
        if instruction is None or path_id is None:
            continue
        lookup[(path_id, normalize_instruction_text(str(instruction)))] = item
    return lookup


def encode_reference_poses(
    annotations: list[dict[str, Any]],
    *,
    language: str | None = None,
) -> list[Pose]:
    poses: list[Pose] = []
    for item in annotations:
        if language is not None and item.get("language") != language:
            continue
        instr_id = item.get("instr_id") or item.get("instruction_id") or item.get("path_id")
        if instr_id is None:
            raise KeyError("Missing instruction identifier in reference annotation.")
        poses.append(
            Pose(
                instr_id=str(instr_id),
                scan=str(item["scan"]),
                viewpoint=_last_viewpoint(item),
                heading=_final_heading(item),
                elevation=_first_present(item, ("elevation", "final_elevation", "end_elevation")),
                path_id=item.get("path_id"),
                language=item.get("language"),
                instruction=item.get("instruction"),
            )
        )
    return poses


def encode_prediction_poses(
    predictions: list[dict[str, Any]],
    *,
    reference_lookup: dict[tuple[Any, str], dict[str, Any]] | None = None,
) -> list[Pose]:
    poses: list[Pose] = []
    for item in predictions:
        tail = _final_step(item)
        instructions = item.get("instructions")
        instruction_text = str(instructions[0]) if isinstance(instructions, list) and instructions else None
        reference_item = None
        if reference_lookup is not None and instruction_text is not None and item.get("path_id") is not None:
            key = (item.get("path_id"), normalize_instruction_text(instruction_text))
            reference_item = reference_lookup.get(key)

        instr_id = item.get("instr_id") or item.get("instruction_id")
        if instr_id is None and reference_item is not None:
            instr_id = reference_item.get("instruction_id") or reference_item.get("instr_id")
        if instr_id is None:
            instr_id = item.get("path_id")
        if instr_id is None:
            raise KeyError("Missing instruction identifier in prediction item.")

        scan = item.get("scan") or tail.get("scan")
        viewpoint = tail.get("viewpoint") or tail.get("vp") or tail.get("image_id")
        if scan is None or viewpoint is None:
            raise KeyError(f"Prediction item {instr_id} is missing scan/viewpoint fields.")

        poses.append(
            Pose(
                instr_id=str(instr_id),
                scan=str(scan),
                viewpoint=str(viewpoint),
                heading=float(tail.get("heading", item.get("heading", 0.0))),
                elevation=float(tail.get("elevation", item.get("elevation", 0.0))),
                path_id=item.get("path_id"),
                language=reference_item.get("language") if reference_item is not None else None,
                instruction=instruction_text,
            )
        )
    return poses


def _last_viewpoint(raw_item: dict[str, Any]) -> str:
    sub_paths = raw_item.get("sub_paths")
    if isinstance(sub_paths, list) and sub_paths:
        tail = sub_paths[-1]
        if isinstance(tail, list) and tail:
            return str(tail[-1])
        return str(tail)

    sub_trajectory = raw_item.get("sub_trajectory") or raw_item.get("trajectory") or raw_item.get("path")
    if isinstance(sub_trajectory, list) and sub_trajectory:
        tail = sub_trajectory[-1]
        if isinstance(tail, list) and tail:
            return str(tail[-1])
        if isinstance(tail, dict):
            value = tail.get("viewpoint") or tail.get("vp") or tail.get("image_id")
            if value is not None:
                return str(value)
        return str(tail)

    for key in ("viewpoint", "vp", "end_viewpoint", "goal_viewpoint"):
        value = raw_item.get(key)
        if value is not None:
            return str(value)
    raise KeyError("Unable to infer final viewpoint from annotation item.")


def _first_present(raw_item: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        value = raw_item.get(key)
        if value is not None:
            return float(value)
    return default


def _final_heading(raw_item: dict[str, Any]) -> float:
    headings = raw_item.get("headings")
    if isinstance(headings, list) and headings:
        return float(headings[-1])
    return _first_present(raw_item, ("heading", "final_heading", "end_heading"))


def _final_step(item: dict[str, Any]) -> dict[str, Any]:
    poses = item.get("poses")
    if isinstance(poses, list) and poses:
        tail_pose = poses[-1]
        if isinstance(tail_pose, dict):
            return tail_pose

    trajectory = item.get("trajectory") or item.get("pred_path") or item.get("path")
    if isinstance(trajectory, list) and trajectory:
        tail = trajectory[-1]
        if isinstance(tail, dict):
            return tail
        if isinstance(tail, list) and tail:
            last = tail[-1]
            if isinstance(last, dict):
                return last
            return {"viewpoint": last}
        return {"viewpoint": tail}
    return item
