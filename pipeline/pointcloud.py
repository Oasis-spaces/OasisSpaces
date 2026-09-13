"""Minimal point-cloud I/O and cleanup utilities (numpy only).

Reads/writes PLY point clouds (binary little-endian and ascii) without
external 3D libraries, plus two cleanup passes used after reconstruction:
voxel downsampling and occupancy-based outlier removal.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


@dataclass
class PointCloud:
    points: np.ndarray  # (N, 3) float32
    colors: np.ndarray  # (N, 3) uint8
    normals: np.ndarray | None = None  # (N, 3) float32 unit vectors, if known
    labels: np.ndarray | None = None  # (N,) uint8 index into label_names
    label_names: list[str] | None = None  # index 0 is always "unlabelled"

    def __len__(self) -> int:
        return len(self.points)

    def subset(self, keep: np.ndarray) -> PointCloud:
        """The points picked by a boolean mask or index array."""
        return PointCloud(self.points[keep], self.colors[keep],
                          None if self.normals is None else self.normals[keep],
                          None if self.labels is None else self.labels[keep],
                          self.label_names)

    def points_labelled(self, name: str) -> np.ndarray:
        """Boolean mask of the points carrying this label."""
        if self.labels is None or not self.label_names or name not in self.label_names:
            return np.zeros(len(self), bool)
        return self.labels == self.label_names.index(name)


def load_ply(path: str | Path) -> PointCloud:
    path = Path(path)
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")

        fmt = None
        vertex_count = 0
        properties: list[tuple[str, str]] = []  # (name, numpy dtype code)
        label_names: list[str] | None = None
        in_vertex_element = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected end of header")
            tokens = line.decode("ascii", "replace").strip().split()
            if not tokens:
                continue
            if tokens[0] == "comment":
                if len(tokens) > 2 and tokens[1] == "label_names":
                    # Names can contain spaces ("chest of drawers"), so the
                    # rest of the line is one comma-separated list.
                    label_names = " ".join(tokens[2:]).split(",")
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                in_vertex_element = tokens[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(tokens[2])
            elif tokens[0] == "property" and in_vertex_element:
                if tokens[1] == "list":
                    raise ValueError(f"{path}: list property in vertex element")
                properties.append((tokens[2], _PLY_TYPES[tokens[1]]))
            elif tokens[0] == "end_header":
                break

        if fmt == "ascii":
            names = [name for name, _ in properties]
            data = np.loadtxt(f, dtype=np.float64, max_rows=vertex_count, ndmin=2)
            columns = {name: data[:, i] for i, name in enumerate(names)}
        elif fmt == "binary_little_endian":
            dtype = np.dtype([(name, "<" + code) for name, code in properties])
            raw = np.frombuffer(f.read(dtype.itemsize * vertex_count), dtype=dtype)
            columns = {name: raw[name] for name, _ in properties}
        else:
            raise ValueError(f"{path}: unsupported PLY format {fmt}")

    points = np.column_stack(
        [columns["x"], columns["y"], columns["z"]]
    ).astype(np.float32)
    if "red" in columns:
        colors = np.column_stack(
            [columns["red"], columns["green"], columns["blue"]]
        )
        if colors.dtype != np.uint8:
            if colors.max() <= 1.0:
                colors = colors * 255.0
            colors = colors.astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    normals = None
    if "nx" in columns:
        normals = np.column_stack(
            [columns["nx"], columns["ny"], columns["nz"]]
        ).astype(np.float32)
    labels = columns["label"].astype(np.uint8) if "label" in columns else None
    return PointCloud(points, colors, normals, labels, label_names)


def save_ply(cloud: PointCloud, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if cloud.normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if cloud.labels is not None:
        fields += [("label", "u1")]
    names_comment = (f"comment label_names {','.join(cloud.label_names)}\n"
                     if cloud.labels is not None and cloud.label_names else "")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(cloud)}\n"
        + names_comment
        + "".join(f"property {'float' if kind == '<f4' else 'uchar'} {name}\n"
                  for name, kind in fields)
        + "end_header\n"
    )
    data = np.empty(len(cloud), dtype=np.dtype(fields))
    points = cloud.points.astype(np.float32)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    if cloud.normals is not None:
        normals = cloud.normals.astype(np.float32)
        data["nx"], data["ny"], data["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    colors = cloud.colors.astype(np.uint8)
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    if cloud.labels is not None:
        data["label"] = cloud.labels.astype(np.uint8)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def trim_far_points(cloud: PointCloud, factor: float = 10.0) -> PointCloud:
    """Drop far-field junk: points beyond `factor` x the median distance
    from the cloud's median center. SfM triangulates a few wildly distant
    points from mismatches; they wreck bounding-box-based viewers."""
    if len(cloud) < 10:
        return cloud
    center = np.median(cloud.points, axis=0)
    dist = np.linalg.norm(cloud.points - center, axis=1)
    cutoff = factor * float(np.median(dist))
    if cutoff <= 0:
        return cloud
    keep = dist <= cutoff
    return cloud.subset(keep)


def _voxel_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    grid = np.floor(points / voxel_size).astype(np.int64)
    grid -= grid.min(axis=0)
    dims = grid.max(axis=0) + 1
    return (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    """Keep one representative point per voxel (the first seen)."""
    if len(cloud) == 0 or voxel_size <= 0:
        return cloud
    keys = _voxel_indices(cloud.points, voxel_size)
    _, keep = np.unique(keys, return_index=True)
    return cloud.subset(keep)


def remove_outliers(
    cloud: PointCloud, voxel_size: float, min_neighbors: int = 3
) -> PointCloud:
    """Drop points in sparsely occupied voxels.

    A point survives if its voxel plus the 26 surrounding voxels together
    hold at least `min_neighbors` other points — a cheap stand-in for
    radius-based outlier removal that stays pure numpy.
    """
    if len(cloud) == 0 or voxel_size <= 0:
        return cloud
    grid = np.floor(cloud.points / voxel_size).astype(np.int64)
    grid -= grid.min(axis=0)
    dims = grid.max(axis=0) + 2
    keys = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )

    neighbor_counts = np.zeros(len(unique_keys), dtype=np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                offset = (dx * dims[1] + dy) * dims[2] + dz
                shifted = unique_keys + offset
                pos = np.searchsorted(unique_keys, shifted)
                pos_clipped = np.minimum(pos, len(unique_keys) - 1)
                hit = unique_keys[pos_clipped] == shifted
                neighbor_counts[hit] += counts[pos_clipped[hit]]

    keep = (neighbor_counts[inverse] - 1) >= min_neighbors
    return cloud.subset(keep)
