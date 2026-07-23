"""Shared data models for pose encoding and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Pose:
    instr_id: str
    scan: str
    viewpoint: str
    heading: float
    elevation: float
    x: float | None = None
    y: float | None = None
    z: float | None = None
    path_id: Any = None
    language: str | None = None
    instruction: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Pose":
        return cls(
            instr_id=str(record["instr_id"]),
            scan=str(record["scan"]),
            viewpoint=str(record["viewpoint"]),
            heading=float(record["heading"]),
            elevation=float(record["elevation"]),
            x=_optional_float(record.get("x")),
            y=_optional_float(record.get("y")),
            z=_optional_float(record.get("z")),
            path_id=record.get("path_id"),
            language=record.get("language"),
            instruction=record.get("instruction"),
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "instr_id": self.instr_id,
            "scan": self.scan,
            "viewpoint": self.viewpoint,
            "heading": self.heading,
            "elevation": self.elevation,
        }
        if self.x is not None and self.y is not None and self.z is not None:
            record["x"] = self.x
            record["y"] = self.y
            record["z"] = self.z
        if self.path_id is not None:
            record["path_id"] = self.path_id
        if self.language is not None:
            record["language"] = self.language
        if self.instruction is not None:
            record["instruction"] = self.instruction
        return record

    @property
    def has_position(self) -> bool:
        return self.x is not None and self.y is not None and self.z is not None

    @property
    def position(self) -> tuple[float, float, float]:
        if not self.has_position:
            raise ValueError(f"Pose {self.instr_id} is missing x/y/z coordinates.")
        return (float(self.x), float(self.y), float(self.z))


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    hfov: float
    vfov: float
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(slots=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def to_record(self) -> dict[str, float]:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
        }


@dataclass(slots=True)
class ProjectionResult:
    raw_bbox: BoundingBox
    bbox: BoundingBox
    visible_ratio: float


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

