from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .common import GRID_H, GRID_W, IMAGE_SIZE


@dataclass(frozen=True)
class MeshPair:
    src_vertices: np.ndarray
    dst_vertices1: np.ndarray
    dst_vertices2: np.ndarray
    image_width: int = IMAGE_SIZE
    image_height: int = IMAGE_SIZE
    grid_w: int = GRID_W
    grid_h: int = GRID_H


def identity_vertices(width: int = IMAGE_SIZE, height: int = IMAGE_SIZE) -> np.ndarray:
    xs = np.linspace(0.0, float(width), GRID_W + 1)
    ys = np.linspace(0.0, float(height), GRID_H + 1)
    vertices = []
    for y in ys:
        for x in xs:
            vertices.append([x, y])
    return np.asarray(vertices, dtype=np.float64)


def build_mesh_pair(mesh: np.ndarray) -> MeshPair:
    src = identity_vertices()
    predicted = np.asarray(mesh, dtype=np.float64).reshape(-1, 2)
    return MeshPair(src_vertices=src, dst_vertices1=src.copy(), dst_vertices2=predicted)


def find_feature_matches(image1_bgr: np.ndarray, image2_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
        ratio = 0.75
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        if desc1 is None or desc2 is None:
            raise ValueError("SIFT did not find descriptors")
        matcher = cv2.BFMatcher(norm)
        raw = matcher.knnMatch(desc1, desc2, k=2)
        matches = [m for m, n in raw if m.distance < ratio * n.distance]
    else:
        detector = cv2.ORB_create(5000)
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        if desc1 is None or desc2 is None:
            raise ValueError("ORB did not find descriptors")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(desc1, desc2), key=lambda m: m.distance)[:1000]

    if len(matches) < 4:
        raise ValueError(f"not enough feature matches: {len(matches)}")

    pts1 = np.asarray([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts2 = np.asarray([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)
    return pts1, pts2


def compute_mesh_rmse(pts1: np.ndarray, pts2: np.ndarray, meshes: MeshPair) -> float:
    polygons = _polygon_indices(meshes.grid_w, meshes.grid_h)
    rmse_sum = 0.0
    feature_num = 0
    for p1, p2 in zip(pts1, pts2):
        idx1 = _grid_index(p1, meshes.image_width, meshes.image_height, meshes.grid_w, meshes.grid_h)
        idx2 = _grid_index(p2, meshes.image_width, meshes.image_height, meshes.grid_w, meshes.grid_h)
        poly1 = polygons[idx1]
        poly2 = polygons[idx2]
        verify1 = _verify_vertex_index(p1, poly1, meshes.src_vertices)
        verify2 = _verify_vertex_index(p2, poly2, meshes.src_vertices)
        affine1 = _affine_transform(meshes.src_vertices, meshes.dst_vertices1, poly1, verify1)
        affine2 = _affine_transform(meshes.src_vertices, meshes.dst_vertices2, poly2, verify2)
        warped1 = _apply_affine(affine1, p1)
        warped2 = _apply_affine(affine2, p2)
        rmse_sum += float(np.linalg.norm(warped1 - warped2))
        feature_num += 1

    if feature_num == 0:
        raise ValueError("no valid feature matches for mesh RMSE")
    return float(math.sqrt(rmse_sum / feature_num))


def compute_warping_residual(meshes: MeshPair) -> tuple[float, float]:
    avg, sd = _mesh_line_residual(meshes.dst_vertices2, meshes.grid_w, meshes.grid_h)
    return avg, sd


def _polygon_indices(nw: int, nh: int) -> list[list[int]]:
    polygons = []
    for h in range(nh):
        for w in range(nw):
            polygons.append(
                [
                    w + h * (nw + 1),
                    (w + 1) + h * (nw + 1),
                    (w + 1) + (h + 1) * (nw + 1),
                    w + (h + 1) * (nw + 1),
                ]
            )
    return polygons


def _grid_index(point: np.ndarray, width: int, height: int, nw: int, nh: int) -> int:
    gx = int(point[0] / (width / float(nw)))
    gy = int(point[1] / (height / float(nh)))
    gx = max(0, min(nw - 1, gx))
    gy = max(0, min(nh - 1, gy))
    return gx + gy * nw


def _verify_vertex_index(point: np.ndarray, polygon: list[int], vertices: np.ndarray) -> int:
    v1 = vertices[polygon[1]]
    v3 = vertices[polygon[3]]
    return polygon[3] if float(np.sum((point - v1) ** 2)) > float(np.sum((point - v3) ** 2)) else polygon[1]


def _affine_transform(src_vertices: np.ndarray, dst_vertices: np.ndarray, polygon: list[int], verify_index: int):
    src_tri = np.float32([src_vertices[polygon[0]], src_vertices[polygon[2]], src_vertices[verify_index]])
    dst_tri = np.float32([dst_vertices[polygon[0]], dst_vertices[polygon[2]], dst_vertices[verify_index]])
    return cv2.getAffineTransform(src_tri, dst_tri)


def _apply_affine(affine: np.ndarray, point: np.ndarray) -> np.ndarray:
    return affine @ np.array([point[0], point[1], 1.0], dtype=np.float64)


def _mesh_line_residual(vertices: np.ndarray, nw: int, nh: int) -> tuple[float, float]:
    rows = []
    cols = []
    for row_index in range(nh + 1):
        rows.append(np.asarray([vertices[w + row_index * (nw + 1)] for w in range(nw + 1)], dtype=np.float64))
    for col_index in range(nw + 1):
        cols.append(np.asarray([vertices[col_index + row * (nw + 1)] for row in range(nh + 1)], dtype=np.float64))

    avgs = []
    sds = []
    for points in rows + cols:
        avg, sd = _line_residual(points)
        avgs.append(avg)
        sds.append(sd)
    return float(np.mean(avgs)), float(np.mean(sds))


def _line_residual(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] < 2:
        return 0.0, 0.0
    vx, vy, x0, y0 = cv2.fitLine(np.asarray(points, dtype=np.float32), cv2.DIST_L2, 0, 1e-2, 1e-2).reshape(-1)
    if abs(float(vx)) < 1e-12:
        a, b, c = 1.0, 0.0, -float(x0)
    else:
        a = float(vy / vx)
        b = -1.0
        c = float(y0 - a * x0)
    denom = math.hypot(a, b)
    residuals = np.abs(a * points[:, 0] + b * points[:, 1] + c) / denom
    return float(np.sqrt(np.mean(residuals * residuals))), float(np.std(residuals))
