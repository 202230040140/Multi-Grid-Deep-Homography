"""Run MGDH + a classical canvas renderer on StitchBench General pairs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .common import (
    CODES_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_OUT_ROOT,
    GRID_H,
    GRID_W,
    IMAGE_SIZE,
    METRICS_FIELDS,
    METHOD_KEY,
    METHOD_VARIANT,
    load_bgr,
    normalize_for_mgdh,
    parse_image_paths,
    read_manifest,
    select_rows,
    write_csv,
    write_json,
)
from .tf_compat import install_tf1_compat, purge_legacy_modules


class MGDHPredictor:
    def __init__(self, checkpoint: Path, gpu: str, compat_mode: str = "auto") -> None:
        self.checkpoint = checkpoint
        self.gpu = gpu
        self.compat_mode = compat_mode
        self.tf = self._load_tensorflow()
        self._build_graph()

    def _load_tensorflow(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        if str(CODES_DIR) not in sys.path:
            sys.path.insert(0, str(CODES_DIR))

        if self.compat_mode == "legacy":
            import tensorflow as tf

            return tf
        if self.compat_mode == "shim":
            purge_legacy_modules()
            return install_tf1_compat()

        try:
            import tensorflow as tf

            purge_legacy_modules()
            __import__("models")
            return tf
        except Exception as exc:
            print(f"[mgdh] Legacy TensorFlow import failed, retrying with TF1 shim: {exc}")
            purge_legacy_modules()
            return install_tf1_compat()

    def _build_graph(self) -> None:
        from models import H_estimator

        tf = self.tf
        self.input_tensor = tf.placeholder(shape=[1, IMAGE_SIZE, IMAGE_SIZE, 6], dtype=tf.float32)
        test_depth = tf.ones_like(self.input_tensor[..., 0:1])
        with tf.variable_scope("generator", reuse=None):
            (
                _warp_depth,
                self.mesh_tensor,
                _warp_h1,
                _warp_h2,
                self.warp_tensor,
                _mask_h1,
                _mask_h2,
                self.mask_tensor,
            ) = H_estimator(self.input_tensor, self.input_tensor, test_depth)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
        saver.restore(self.session, str(self.checkpoint))

    def predict(
        self,
        image1_path: Path,
        image2_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        image1_resized = load_bgr(image1_path)
        image2_resized = load_bgr(image2_path)
        clip = np.concatenate([normalize_for_mgdh(image1_resized), normalize_for_mgdh(image2_resized)], axis=2)
        clip = np.expand_dims(clip, axis=0)
        (mesh,) = self.session.run(
            [self.mesh_tensor],
            feed_dict={self.input_tensor: clip},
        )
        image1_full = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
        image2_full = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)
        if image1_full is None:
            raise FileNotFoundError(f"could not read image: {image1_path}")
        if image2_full is None:
            raise FileNotFoundError(f"could not read image: {image2_path}")
        mgdh_warp, _mgdh_mask = self._full_resolution_blend(image1_full, image2_full, mesh[0])
        try:
            panorama, _full_mask = self._feature_stitch(image1_full, image2_full)
            renderer = "feature_homography"
        except RuntimeError as exc:
            panorama = mgdh_warp
            renderer = f"mgdh_dense_fallback: {exc}"
        return panorama, mesh[0], renderer

    @staticmethod
    def _full_resolution_blend(
        image1: np.ndarray,
        image2: np.ndarray,
        mesh_512: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]
        src_mesh = np.asarray(mesh_512, dtype=np.float32).copy()
        src_mesh[..., 0] *= width2 / float(IMAGE_SIZE)
        src_mesh[..., 1] *= height2 / float(IMAGE_SIZE)
        dst_mesh = _regular_mesh(width1, height1)

        map_x = np.full((height1, width1), -1.0, dtype=np.float32)
        map_y = np.full((height1, width1), -1.0, dtype=np.float32)
        for y in range(GRID_H):
            for x in range(GRID_W):
                src_quad = np.float32(
                    [
                        src_mesh[y, x],
                        src_mesh[y, x + 1],
                        src_mesh[y + 1, x + 1],
                        src_mesh[y + 1, x],
                    ]
                )
                dst_quad = np.float32(
                    [
                        dst_mesh[y, x],
                        dst_mesh[y, x + 1],
                        dst_mesh[y + 1, x + 1],
                        dst_mesh[y + 1, x],
                    ]
                )
                if cv2.contourArea(src_quad) <= 1e-3 or cv2.contourArea(dst_quad) <= 1e-3:
                    continue
                transform = cv2.getPerspectiveTransform(dst_quad, src_quad)
                x0, y0 = np.floor(dst_quad.min(axis=0)).astype(np.int32)
                x1, y1 = np.ceil(dst_quad.max(axis=0)).astype(np.int32)
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(width1, x1)
                y1 = min(height1, y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                ones = np.ones_like(xx)
                coords = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3)
                src = coords @ transform.T
                denom = src[:, 2]
                valid = np.abs(denom) > 1e-6
                src_xy = np.full((coords.shape[0], 2), -1.0, dtype=np.float32)
                src_xy[valid, 0] = src[valid, 0] / denom[valid]
                src_xy[valid, 1] = src[valid, 1] / denom[valid]
                map_x[y0:y1, x0:x1] = src_xy[:, 0].reshape(y1 - y0, x1 - x0)
                map_y[y0:y1, x0:x1] = src_xy[:, 1].reshape(y1 - y0, x1 - x0)

        valid_mask = (map_x >= 0.0) & (map_x <= width2 - 1.0) & (map_y >= 0.0) & (map_y <= height2 - 1.0)
        remap_x = np.where(valid_mask, map_x, 0.0).astype(np.float32)
        remap_y = np.where(valid_mask, map_y, 0.0).astype(np.float32)
        warped2 = cv2.remap(
            image2,
            remap_x,
            remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_mask = (valid_mask.astype(np.uint8) * 255)

        panorama = image1.copy()
        overlap = valid_mask
        if np.any(overlap):
            alpha = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=max(width1, height1) / 200.0)
            alpha = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
            blended = (1.0 - 0.5 * alpha) * image1.astype(np.float32) + (0.5 * alpha) * warped2.astype(np.float32)
            panorama[overlap] = np.clip(blended[overlap], 0, 255).astype(np.uint8)
        return panorama, warped_mask

    @staticmethod
    def _feature_stitch(image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        homography, inliers = _estimate_feature_homography(image1, image2)
        if inliers < 12:
            raise RuntimeError(f"insufficient homography inliers: {inliers}")

        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]
        corners1 = np.float32([[0, 0], [width1, 0], [width1, height1], [0, height1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [width2, 0], [width2, height2], [0, height2]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, homography)
        all_corners = np.concatenate([corners1, warped_corners2], axis=0).reshape(-1, 2)
        if not np.isfinite(all_corners).all():
            raise RuntimeError("homography produced non-finite canvas bounds")

        mins = np.floor(all_corners.min(axis=0)).astype(np.int32)
        maxs = np.ceil(all_corners.max(axis=0)).astype(np.int32)
        out_width = int(maxs[0] - mins[0])
        out_height = int(maxs[1] - mins[1])
        max_reasonable_width = int((width1 + width2) * 3)
        max_reasonable_height = int((height1 + height2) * 3)
        if out_width <= 0 or out_height <= 0 or out_width > max_reasonable_width or out_height > max_reasonable_height:
            raise RuntimeError(f"unreasonable canvas size: {out_width}x{out_height}")

        translation = np.array([[1.0, 0.0, -mins[0]], [0.0, 1.0, -mins[1]], [0.0, 0.0, 1.0]])
        warp_matrix = translation @ homography
        warped2 = cv2.warpPerspective(
            image2,
            warp_matrix,
            (out_width, out_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask2 = cv2.warpPerspective(
            np.full((height2, width2), 255, dtype=np.uint8),
            warp_matrix,
            (out_width, out_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

        canvas1 = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        mask1 = np.zeros((out_height, out_width), dtype=np.uint8)
        x0 = int(-mins[0])
        y0 = int(-mins[1])
        canvas1[y0 : y0 + height1, x0 : x0 + width1] = image1
        mask1[y0 : y0 + height1, x0 : x0 + width1] = 255

        panorama = canvas1.copy()
        only2 = (mask2 > 0) & (mask1 == 0)
        overlap = (mask1 > 0) & (mask2 > 0)
        panorama[only2] = warped2[only2]
        if np.any(overlap):
            dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 3).astype(np.float32)
            dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3).astype(np.float32)
            alpha = (dist2 / (dist1 + dist2 + 1e-6))[..., None]
            blended = canvas1.astype(np.float32) * (1.0 - alpha) + warped2.astype(np.float32) * alpha
            panorama[overlap] = np.clip(blended[overlap], 0, 255).astype(np.uint8)

        union_mask = (((mask1 > 0) | (mask2 > 0)).astype(np.uint8) * 255)
        return panorama, union_mask

    def close(self) -> None:
        self.session.close()


def _regular_mesh(width: int, height: int) -> np.ndarray:
    xs = np.linspace(0.0, float(width), GRID_W + 1, dtype=np.float32)
    ys = np.linspace(0.0, float(height), GRID_H + 1, dtype=np.float32)
    mesh = np.zeros((GRID_H + 1, GRID_W + 1, 2), dtype=np.float32)
    for y, yy in enumerate(ys):
        for x, xx in enumerate(xs):
            mesh[y, x] = [xx, yy]
    return mesh


def _estimate_feature_homography(image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, int]:
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    try:
        detector = cv2.SIFT_create(nfeatures=6000)
        norm = cv2.NORM_L2
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=10000)
        norm = cv2.NORM_HAMMING

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("feature detector produced no descriptors")

    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(descriptors2, descriptors1, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)
    if len(good_matches) < 12:
        raise RuntimeError(f"insufficient feature matches: {len(good_matches)}")

    points2 = np.float32([keypoints2[match.queryIdx].pt for match in good_matches])
    points1 = np.float32([keypoints1[match.trainIdx].pt for match in good_matches])
    homography, inlier_mask = cv2.findHomography(points2, points1, cv2.RANSAC, 4.0)
    if homography is None or inlier_mask is None:
        raise RuntimeError("RANSAC homography failed")
    return homography, int(inlier_mask.sum())


def _merge_metrics_rows(out_root: Path, manifest_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics_path = out_root / "metrics.csv"
    by_scene: dict[str, dict[str, str]] = {}
    if metrics_path.exists():
        import csv

        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                by_scene[row["scene"]] = row
    for row in new_rows:
        by_scene[row["scene"]] = row

    ordered = []
    for index, row in enumerate(manifest_rows, start=1):
        scene = row["dataset"]
        if scene not in by_scene:
            continue
        merged = {key: by_scene[scene].get(key, "") for key in METRICS_FIELDS}
        merged["index"] = str(index)
        ordered.append(merged)
    return ordered


def run(args: argparse.Namespace) -> list[dict[str, str]]:
    manifest_rows = read_manifest(Path(args.manifest))
    rows = select_rows(manifest_rows, scenes=args.scene, limit=args.limit)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    predictor: MGDHPredictor | None = None
    metrics_rows: list[dict[str, str]] = []
    try:
        pending = []
        for row in rows:
            scene = row["dataset"]
            pano_path = out_root / scene / "panorama.png"
            if args.skip_existing and not args.force and pano_path.exists():
                pending.append((row, True))
            else:
                pending.append((row, False))
        if any(not skipped for _row, skipped in pending):
            predictor = MGDHPredictor(Path(args.checkpoint), args.gpu, args.compat_mode)

        for index, (row, skipped) in enumerate(pending, start=1):
            scene = row["dataset"]
            scene_dir = out_root / scene
            pano_path = scene_dir / "panorama.png"
            left, right, left_name, right_name = parse_image_paths(row)
            print(f"[{index}/{len(rows)}] {scene}")

            if skipped:
                metrics = {
                    "index": index,
                    "scene": scene,
                    "img1": left_name,
                    "img2": right_name,
                    "status": "ok",
                    "runtime_seconds": "",
                    "panorama": "panorama.png",
                    "renderer": "unknown",
                    "error": "",
                }
                prior = scene_dir / "metrics.json"
                if prior.exists():
                    loaded = json.loads(prior.read_text(encoding="utf-8"))
                    loaded["index"] = index
                    metrics.update({key: loaded.get(key, metrics.get(key, "")) for key in METRICS_FIELDS})
                metrics_rows.append(metrics)
                print("  -> skip existing")
                continue

            started = time.perf_counter()
            try:
                if predictor is None:
                    raise RuntimeError("MGDH predictor was not initialized")
                scene_dir.mkdir(parents=True, exist_ok=True)
                panorama, mesh, renderer = predictor.predict(left, right)
                cv2.imwrite(str(pano_path), panorama)
                np.save(scene_dir / "mesh.npy", mesh)
                runtime = time.perf_counter() - started
                metrics = {
                    "index": index,
                    "scene": scene,
                    "img1": left_name,
                    "img2": right_name,
                    "status": "ok",
                    "runtime_seconds": f"{runtime:.2f}",
                    "panorama": "panorama.png",
                    "renderer": renderer,
                    "error": "",
                }
                write_json(scene_dir / "metrics.json", metrics)
                metrics_rows.append(metrics)
                print(f"  -> ok ({runtime:.1f}s, {renderer})")
            except Exception as exc:
                runtime = time.perf_counter() - started
                error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                metrics = {
                    "index": index,
                    "scene": scene,
                    "img1": left_name,
                    "img2": right_name,
                    "status": "failed",
                    "runtime_seconds": f"{runtime:.2f}",
                    "panorama": "",
                    "renderer": "",
                    "error": error,
                }
                write_json(scene_dir / "metrics.json", metrics)
                metrics_rows.append(metrics)
                print(f"  -> failed: {error}")
                traceback.print_exc()
    finally:
        if predictor is not None:
            predictor.close()

    if args.scene or args.limit > 0:
        metrics_rows = _merge_metrics_rows(out_root, manifest_rows, metrics_rows)
    write_csv(out_root / "metrics.csv", metrics_rows, METRICS_FIELDS)
    summary = {
        "method": METHOD_KEY,
        "method_variant": METHOD_VARIANT,
        "out_root": str(out_root),
        "total": len(metrics_rows),
        "ok": sum(1 for row in metrics_rows if row["status"] == "ok"),
        "failed": sum(1 for row in metrics_rows if row["status"] == "failed"),
        "gpu": args.gpu,
        "checkpoint": args.checkpoint,
    }
    write_json(out_root / "summary.json", summary)

    if not args.skip_eval:
        from .evaluate_stitchbench_mdr_niqe import evaluate_and_write

        evaluate_and_write(
            manifest=Path(args.manifest),
            mgdh_root=out_root,
            output_root=out_root,
            depth_gsp_root=Path(args.depth_gsp_root),
            device=args.device,
            scenes=args.scene,
            limit=args.limit,
            skip_compare=args.skip_compare,
        )
    return metrics_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--depth-gsp-root", default=r"D:\StitchBench_Result\depth_gsp")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compat-mode", choices=["auto", "legacy", "shim"], default="auto")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run(args)
    failed = sum(1 for row in rows if row["status"] == "failed")
    print(f"\nDone. ok={len(rows) - failed}/{len(rows)} metrics={Path(args.out_root) / 'metrics.csv'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
