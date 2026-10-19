"""Track wall boundary extraction for trajectory plots.

The benchmark only ever loads centreline waypoints (x, y, yaw, index) - the
physical track walls exist solely as a Gazebo collision mesh
(f1tenth_gazebo/meshes/<track>.stl), referenced by the matching world file
(f1tenth_gazebo/worlds/<track>.sdf) with a millimetre-to-metre scale and a
model pose. There is no first-class 2D boundary polyline anywhere in the
codebase.

To draw a real boundary (rather than guessing an offset from the centreline)
this module slices the collision mesh with a horizontal plane through the
wall height and keeps every triangle edge that crosses it. Because the walls
are near-vertical extrusions, the unordered set of intersection segments
reconstructs both the inner and outer wall curves without needing to solve
polygon ordering - matplotlib just draws every little segment.

This is a best-effort visual enhancement: any failure to resolve the world
file, mesh file, or a usable slice returns None rather than raising, so a
missing/unusual track never breaks a benchmark run.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ElementTree
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )
except ImportError:  # pragma: no cover - ROS is always present at runtime
    get_package_share_directory = None
    PackageNotFoundError = Exception


_STL_FACE_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attribute_byte_count", "<u2"),
    ]
)


def _read_binary_stl_triangles(path: Path) -> np.ndarray | None:
    """Return an (N, 3, 3) array of triangle vertices, or None if unreadable."""
    data = path.read_bytes()
    header_size = 84
    if len(data) < header_size:
        return None
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = header_size + triangle_count * _STL_FACE_DTYPE.itemsize
    if triangle_count == 0 or len(data) < expected_size:
        # Either empty or not a binary STL (e.g. ASCII "solid ..." format).
        return None
    faces = np.frombuffer(
        data, dtype=_STL_FACE_DTYPE, count=triangle_count, offset=header_size
    )
    return np.stack(
        [faces["v0"], faces["v1"], faces["v2"]], axis=1
    ).astype(np.float64)


def _slice_triangles_at_height(
    triangles: np.ndarray, height: float
) -> np.ndarray | None:
    """Intersect every triangle edge with a horizontal plane at `height`.

    Returns an (M, 2, 2) array of XY line segments, unordered.
    """
    above = triangles[:, :, 2] > height
    crosses_plane = (above.sum(axis=1) > 0) & (above.sum(axis=1) < 3)
    segments = []
    for triangle, triangle_above in zip(
        triangles[crosses_plane], above[crosses_plane]
    ):
        points = []
        for i in range(3):
            j = (i + 1) % 3
            z_i, z_j = triangle[i, 2], triangle[j, 2]
            if triangle_above[i] != triangle_above[j]:
                t = (height - z_i) / (z_j - z_i)
                points.append(triangle[i, :2] + t * (triangle[j, :2] - triangle[i, :2]))
        if len(points) == 2:
            segments.append(points)
    if not segments:
        return None
    return np.asarray(segments, dtype=np.float64)


def _parse_pose(pose_text: str | None) -> tuple[float, float, float]:
    if not pose_text:
        return 0.0, 0.0, 0.0
    values = [float(value) for value in pose_text.split()]
    x = values[0] if len(values) > 0 else 0.0
    y = values[1] if len(values) > 1 else 0.0
    yaw = values[5] if len(values) > 5 else 0.0
    return x, y, yaw


def _find_track_mesh_model(world_root: ElementTree.Element) -> ElementTree.Element | None:
    """Return the first <model> that references a mesh geometry."""
    for model in world_root.iter("model"):
        if model.find(".//mesh/uri") is not None:
            return model
    return None


@lru_cache(maxsize=None)
def load_track_boundary_segments(track_name: str) -> np.ndarray | None:
    """Best-effort world-frame XY wall-boundary segments for `track_name`.

    Returns an (M, 2, 2) array of line segments (in the same world frame as
    track waypoints and recorded car positions), or None if the boundary
    cannot be resolved. Never raises. Results are cached per track name.
    """
    if get_package_share_directory is None:
        return None
    try:
        share_directory = Path(get_package_share_directory("f1tenth_gazebo"))
    except PackageNotFoundError:
        return None

    world_path = share_directory / "worlds" / f"{track_name}.sdf"
    if not world_path.is_file():
        return None

    try:
        world_root = ElementTree.parse(world_path).getroot()
    except ElementTree.ParseError:
        return None

    model = _find_track_mesh_model(world_root)
    if model is None:
        return None

    mesh_uri = next((uri.text for uri in model.iter("uri") if uri.text), None)
    if mesh_uri is None:
        return None

    scale_text = next(
        (scale.text for scale in model.iter("scale") if scale.text), "1 1 1"
    )
    try:
        scale = float(scale_text.split()[0])
    except (ValueError, IndexError):
        return None

    pose_element = model.find("pose")
    pose_x, pose_y, yaw = _parse_pose(
        pose_element.text if pose_element is not None else None
    )

    mesh_path = (world_path.parent / mesh_uri).resolve()
    if not mesh_path.is_file():
        return None

    triangles = _read_binary_stl_triangles(mesh_path)
    if triangles is None:
        return None

    z_values = triangles[:, :, 2]
    z_min, z_max = float(z_values.min()), float(z_values.max())
    if z_max - z_min < 1e-6:
        return None
    slice_height = z_min + 0.5 * (z_max - z_min)

    segments = _slice_triangles_at_height(triangles, slice_height)
    if segments is None:
        return None

    segments = segments * scale
    if abs(yaw) > 1e-9:
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        segments = segments @ rotation.T
    segments[:, :, 0] += pose_x
    segments[:, :, 1] += pose_y

    return segments
