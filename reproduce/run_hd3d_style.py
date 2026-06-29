"""Run MGDH + a classical canvas renderer on HD3D-style two-view manifests."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import DEFAULT_CHECKPOINT, METHOD_KEY, METHOD_VARIANT, REPO_ROOT, write_json
from .run_stitchbench_general import MGDHPredictor

GITHUB_ROOT = REPO_ROOT.parent
GES_TOOLS = GITHUB_ROOT / "GES-GSP-Stitching" / "tools"
if str(GES_TOOLS) not in sys.path:
    sys.path.insert(0, str(GES_TOOLS))

from hd3d_metrics import PairPaths, evaluate_pair_paths  # noqa: E402


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]],
    *,
    pairs: list[str] | None = None,
    scenes: list[str] | None = None,
    limit: int = 0,
) -> list[dict[str, str]]:
    selected = rows
    if pairs:
        wanted = set(pairs)
        selected = [row for row in selected if row.get("pair_name") in wanted]
    if scenes:
        wanted = set(scenes)
        selected = [row for row in selected if row.get("scene") in wanted]
    if limit > 0:
        selected = selected[:limit]
    return selected


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def status_payload(
    row: dict[str, str],
    *,
    success: bool,
    runtime_seconds: float,
    raw_path: Path,
    renderer: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "method": METHOD_KEY,
        "method_variant": METHOD_VARIANT,
        "scene": row.get("scene", ""),
        "pair_id": row.get("pair_id", ""),
        "pair_name": row.get("pair_name", ""),
        "success": success,
        "failure_reason": failure_reason,
        "runtime_seconds": runtime_seconds,
        "raw_path": str(raw_path),
        "renderer": renderer,
    }


def build_pair_paths(row: dict[str, str], method_dir: Path, runtime_seconds: float) -> PairPaths:
    return PairPaths(
        pair_name=row["pair_name"],
        final_pair_dir=Path(row["final_pair_dir"]),
        method_dir=method_dir,
        raw_path=method_dir / "raw.png",
        aligned_path=method_dir / "aligned_to_gt.png",
        valid_mask_path=method_dir / "valid_mask.png",
        gt_path=Path(row["gt_path"]),
        cpp_rmse_path=None,
        cpp_residual_path=None,
        runtime_seconds=runtime_seconds,
        status_path=method_dir / "method_status.json",
        metrics_path=method_dir / "metrics.json",
        stdout_path=method_dir / "run.log",
        stderr_path=method_dir / "error.log",
    )


def process_pair(
    row: dict[str, str],
    *,
    predictor: MGDHPredictor,
    method: str,
    device: str,
    skip_niqe: bool,
    skip_lpips: bool,
    force: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    method_dir = Path(row["final_pair_dir"]) / method
    raw_path = method_dir / "raw.png"
    metrics_path = method_dir / "metrics.json"
    if skip_existing and not force and raw_path.exists() and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8-sig"))

    method_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        panorama, mesh, renderer = predictor.predict(Path(row["left_source"]), Path(row["right_source"]))
        if not cv2.imwrite(str(raw_path), panorama):
            raise RuntimeError(f"failed to write raw image: {raw_path}")
        np.save(method_dir / "mesh.npy", mesh)
        runtime = time.perf_counter() - started
        write_json(
            method_dir / "method_status.json",
            status_payload(row, success=True, runtime_seconds=runtime, raw_path=raw_path, renderer=renderer),
        )
        paths = build_pair_paths(row, method_dir, runtime)
        metrics = evaluate_pair_paths(
            paths,
            method=method,
            scene=row["scene"],
            pair_id=row["pair_id"],
            skip_niqe=skip_niqe,
            skip_lpips=skip_lpips,
            device=device,
        )
        metrics["method_variant"] = METHOD_VARIANT
        metrics["renderer"] = renderer
        write_metrics(metrics_path, metrics)
        return metrics
    except Exception as exc:
        runtime = time.perf_counter() - started
        reason = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        write_json(
            method_dir / "method_status.json",
            status_payload(row, success=False, runtime_seconds=runtime, raw_path=raw_path, failure_reason=reason),
        )
        metrics = {
            "scene": row.get("scene", ""),
            "pair_id": row.get("pair_id", ""),
            "pair_name": row.get("pair_name", ""),
            "method": method,
            "method_variant": METHOD_VARIANT,
            "status": "failed",
            "failure_reason": reason,
            "runtime_seconds": runtime,
            "raw_path": str(raw_path),
            "aligned_path": str(method_dir / "aligned_to_gt.png"),
            "valid_mask_path": str(method_dir / "valid_mask.png"),
            "gt_path": row.get("gt_path", ""),
        }
        write_metrics(metrics_path, metrics)
        return metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "success"]

    def mean_value(key: str) -> float:
        values = [float_or_nan(row.get(key)) for row in successes]
        values = [value for value in values if math.isfinite(value)]
        return sum(values) / len(values) if values else math.nan

    return {
        "method": METHOD_KEY,
        "method_variant": METHOD_VARIANT,
        "total": len(rows),
        "successes": len(successes),
        "failures": len(rows) - len(successes),
        "mean_mdr": mean_value("mdr"),
        "mean_niqe": mean_value("niqe"),
        "mean_psnr": mean_value("psnr"),
        "mean_ssim": mean_value("ssim"),
        "mean_lpips": mean_value("lpips"),
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = load_manifest(Path(args.manifest))
    rows = select_rows(manifest, pairs=args.pair, scenes=args.scene, limit=args.limit)
    if not rows:
        raise RuntimeError("no manifest rows selected")

    results: list[dict[str, Any]] = []
    predictor = MGDHPredictor(Path(args.checkpoint), args.gpu, args.compat_mode)
    try:
        for index, row in enumerate(rows, start=1):
            print(f"[{index}/{len(rows)}] {row['pair_name']}")
            try:
                metrics = process_pair(
                    row,
                    predictor=predictor,
                    method=args.method,
                    device=args.device,
                    skip_niqe=args.skip_niqe,
                    skip_lpips=args.skip_lpips,
                    force=args.force,
                    skip_existing=args.skip_existing,
                )
                results.append(metrics)
                print(f"  -> {metrics.get('status')} renderer={metrics.get('renderer', '')}")
                if args.stop_on_error and metrics.get("status") != "success":
                    raise RuntimeError(metrics.get("failure_reason") or "pair failed")
            except Exception as exc:
                print(f"  -> failed: {exc}")
                traceback.print_exc()
                if args.stop_on_error:
                    raise
    finally:
        predictor.close()

    result_root = Path(args.result_root)
    write_metrics(result_root / args.summary_name, summarize(results))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--method", default=METHOD_KEY)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compat-mode", choices=["auto", "legacy", "shim"], default="auto")
    parser.add_argument("--pair", action="append")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-niqe", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--summary-name", default="mgdh_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run(args)
    failures = sum(1 for row in results if row.get("status") != "success")
    print(f"\nDone. success={len(results) - failures}/{len(results)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
