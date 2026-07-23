"""Rigid geometry, intrinsics, and projection helpers."""

from __future__ import annotations

from math import cos, sin, tan
from typing import Iterable

from .schemas import BoundingBox, CameraIntrinsics, Pose, ProjectionResult


def build_intrinsics(width: int, height: int, hfov: float, vfov: float) -> CameraIntrinsics:
    fx = width / (2.0 * tan(hfov / 2.0))
    fy = height / (2.0 * tan(vfov / 2.0))
    return CameraIntrinsics(
        width=width,
        height=height,
        hfov=hfov,
        vfov=vfov,
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def vector_subtract(lhs: Iterable[float], rhs: Iterable[float]) -> tuple[float, float, float]:
    left = list(lhs)
    right = list(rhs)
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def rotation_y(theta: float) -> tuple[tuple[float, float, float], ...]:
    return (
        (cos(theta), 0.0, sin(theta)),
        (0.0, 1.0, 0.0),
        (-sin(theta), 0.0, cos(theta)),
    )


def rotation_x(phi: float) -> tuple[tuple[float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0),
        (0.0, cos(phi), -sin(phi)),
        (0.0, sin(phi), cos(phi)),
    )


def matmul3(lhs: tuple[tuple[float, float, float], ...], rhs: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(lhs[row][k] * rhs[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def matvec3(matrix: tuple[tuple[float, float, float], ...], vector: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = vector
    return tuple(sum(matrix[row][col] * value for col, value in enumerate((x, y, z))) for row in range(3))


def transpose3(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[col][row] for col in range(3)) for row in range(3))


def camera_rotation(heading: float, elevation: float) -> tuple[tuple[float, float, float], ...]:
    return matmul3(rotation_x(elevation), rotation_y(heading))


def world_to_camera(point_world: Iterable[float], pose: Pose) -> tuple[float, float, float]:
    rotation = camera_rotation(pose.heading, pose.elevation)
    relative = vector_subtract(point_world, pose.position)
    return matvec3(rotation, relative)


def camera_forward_world(heading: float, elevation: float) -> tuple[float, float, float]:
    rotation = camera_rotation(heading, elevation)
    return matvec3(transpose3(rotation), (0.0, 0.0, 1.0))


def landmark_center_from_reference(reference_pose: Pose, distance: float = 1.0) -> tuple[float, float, float]:
    fx, fy, fz = camera_forward_world(reference_pose.heading, reference_pose.elevation)
    x, y, z = reference_pose.position
    return (x + distance * fx, y + distance * fy, z + distance * fz)


def project_camera_point(point_camera: tuple[float, float, float], intrinsics: CameraIntrinsics) -> tuple[float, float]:
    x, y, z = point_camera
    if z <= 0.0:
        raise ValueError("Point is behind the camera and cannot be projected.")
    u = intrinsics.fx * x / z + intrinsics.cx
    v = intrinsics.fy * y / z + intrinsics.cy
    return (u, v)


def build_bbox(center_u: float, center_v: float, alpha_h: float, alpha_v: float, intrinsics: CameraIntrinsics) -> BoundingBox:
    du = intrinsics.fx * tan(alpha_h)
    dv = intrinsics.fy * tan(alpha_v)
    return BoundingBox(
        xmin=center_u - du,
        ymin=center_v - dv,
        xmax=center_u + du,
        ymax=center_v + dv,
    )


def clip_bbox(bbox: BoundingBox, width: int, height: int) -> BoundingBox:
    return BoundingBox(
        xmin=max(0.0, min(float(width), bbox.xmin)),
        ymin=max(0.0, min(float(height), bbox.ymin)),
        xmax=max(0.0, min(float(width), bbox.xmax)),
        ymax=max(0.0, min(float(height), bbox.ymax)),
    )


def visibility_ratio(bbox: BoundingBox, width: int, height: int) -> float:
    if bbox.area <= 0.0:
        return 0.0
    clipped = clip_bbox(bbox, width, height)
    return clipped.area / bbox.area


def project_landmark(
    point_world: tuple[float, float, float],
    pose: Pose,
    intrinsics: CameraIntrinsics,
    alpha_h: float,
    alpha_v: float,
) -> ProjectionResult:
    point_camera = world_to_camera(point_world, pose)
    center_u, center_v = project_camera_point(point_camera, intrinsics)
    raw_bbox = build_bbox(center_u, center_v, alpha_h, alpha_v, intrinsics)
    clipped_bbox = clip_bbox(raw_bbox, intrinsics.width, intrinsics.height)
    return ProjectionResult(
        raw_bbox=raw_bbox,
        bbox=clipped_bbox,
        visible_ratio=visibility_ratio(raw_bbox, intrinsics.width, intrinsics.height),
    )

