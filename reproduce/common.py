from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CODES_DIR = REPO_ROOT / "Codes"

DEFAULT_MANIFEST = r"D:\StitchBench_Result\_global_work\_shared\manifest.csv"
DEFAULT_OUT_ROOT = r"D:\StitchBench_Result\mgdh"
DEFAULT_DEPTH_GSP_ROOT = r"D:\StitchBench_Result\depth_gsp"
DEFAULT_CHECKPOINT = str(CODES_DIR / "checkpoints" / "model.ckpt-500000")
METHOD_KEY = "mgdh"
METHOD_VARIANT = "MGDH checkpoint + classical feature-homography canvas renderer"
IMAGE_SIZE = 512
GRID_W = 8
GRID_H = 8

METRICS_FIELDS = [
    "index",
    "scene",
    "img1",
    "img2",
    "status",
    "runtime_seconds",
    "panorama",
    "renderer",
    "error",
]

PER_PAIR_FIELDS = [
    "dataset",
    "category",
    "result_image",
    "mdr_rmse",
    "warping_residual_avg",
    "warping_residual_sd",
    "niqe",
    "status",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]],
    *,
    scenes: list[str] | None = None,
    limit: int = 0,
) -> list[dict[str, str]]:
    selected = rows
    if scenes:
        wanted = set(scenes)
        selected = [row for row in selected if row.get("dataset") in wanted]
    if limit > 0:
        selected = selected[:limit]
    return selected


def parse_image_paths(row: dict[str, str]) -> tuple[Path, Path, str, str]:
    image_files = (row.get("image_files") or "").split("|")
    if len(image_files) < 2:
        raise ValueError(f"expected two input images for {row.get('dataset', '<unknown>')}")
    data_dir = Path(row["data_dir"])
    left_name = image_files[0]
    right_name = image_files[1]
    return data_dir / left_name, data_dir / right_name, left_name, right_name


def load_bgr(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def normalize_for_mgdh(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr.astype(np.float32) / 127.5 - 1.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.5f}"
