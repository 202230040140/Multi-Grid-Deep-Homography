"""Evaluate MGDH StitchBench outputs with MDR/NIQE tables."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .common import (
    DEFAULT_DEPTH_GSP_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUT_ROOT,
    METHOD_VARIANT,
    PER_PAIR_FIELDS,
    fmt_float,
    load_bgr,
    parse_image_paths,
    read_manifest,
    select_rows,
    write_csv,
)
from .mesh_metrics import build_mesh_pair, compute_mesh_rmse, compute_warping_residual, find_feature_matches


def parse_float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def compute_niqe(image_path: Path, device: str) -> float:
    import pyiqa

    metric = pyiqa.create_metric("niqe", device=device)
    score = metric(str(image_path))
    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def evaluate_rows(
    rows: list[dict[str, str]],
    *,
    mgdh_root: Path,
    device: str,
    skip_mdr: bool = False,
    skip_niqe: bool = False,
) -> list[dict[str, Any]]:
    output_rows = []
    niqe_metric = None
    if not skip_niqe:
        try:
            import pyiqa

            niqe_metric = pyiqa.create_metric("niqe", device=device)
        except Exception as exc:
            print(f"[mgdh-eval] NIQE metric unavailable: {exc}")

    for row in rows:
        dataset = row["dataset"]
        category = row.get("category", "")
        scene_dir = mgdh_root / dataset
        panorama = scene_dir / "panorama.png"
        mesh_path = scene_dir / "mesh.npy"
        mdr = residual_avg = residual_sd = niqe = math.nan
        status = "ok"

        if not panorama.exists():
            status = "missing_result"
        else:
            if not skip_mdr:
                try:
                    if not mesh_path.exists():
                        raise FileNotFoundError(f"missing mesh: {mesh_path}")
                    left, right, _left_name, _right_name = parse_image_paths(row)
                    image1 = load_bgr(left)
                    image2 = load_bgr(right)
                    pts1, pts2 = find_feature_matches(image1, image2)
                    meshes = build_mesh_pair(np.load(mesh_path))
                    mdr = compute_mesh_rmse(pts1, pts2, meshes)
                    residual_avg, residual_sd = compute_warping_residual(meshes)
                except Exception as exc:
                    print(f"[mgdh-eval] MDR failed for {dataset}: {exc}")
                    status = "failed"
            if niqe_metric is not None:
                try:
                    score = niqe_metric(str(panorama))
                    niqe = float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)
                except Exception as exc:
                    print(f"[mgdh-eval] NIQE failed for {dataset}: {exc}")
                    status = "failed"
            elif not skip_niqe:
                status = "failed"

        output_rows.append(
            {
                "dataset": dataset,
                "category": category,
                "result_image": str(panorama),
                "mdr_rmse": fmt_float(mdr),
                "warping_residual_avg": fmt_float(residual_avg),
                "warping_residual_sd": fmt_float(residual_sd),
                "niqe": fmt_float(niqe),
                "status": status,
            }
        )
    return output_rows


def write_by_category(output_root: Path, rows: list[dict[str, Any]]) -> None:
    categories = sorted({row.get("category", "") for row in rows})
    out_rows = []
    for category in categories:
        group = [row for row in rows if row.get("category", "") == category]
        mdr_values = [parse_float(row.get("mdr_rmse")) for row in group]
        niqe_values = [parse_float(row.get("niqe")) for row in group]
        out_rows.append(
            {
                "category": category,
                "total_count": len(group),
                "valid_mdr_count": sum(1 for value in mdr_values if math.isfinite(value)),
                "valid_niqe_count": sum(1 for value in niqe_values if math.isfinite(value)),
                "mdr_rmse_mean": fmt_float(finite_mean(mdr_values)),
                "warping_residual_avg_mean": fmt_float(
                    finite_mean([parse_float(row.get("warping_residual_avg")) for row in group])
                ),
                "warping_residual_sd_mean": fmt_float(
                    finite_mean([parse_float(row.get("warping_residual_sd")) for row in group])
                ),
                "niqe_mean": fmt_float(finite_mean(niqe_values)),
            }
        )
    write_csv(
        output_root / "by_category.csv",
        out_rows,
        [
            "category",
            "total_count",
            "valid_mdr_count",
            "valid_niqe_count",
            "mdr_rmse_mean",
            "warping_residual_avg_mean",
            "warping_residual_sd_mean",
            "niqe_mean",
        ],
    )


def load_per_pair(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["dataset"]: row for row in csv.DictReader(handle)}


def merge_per_pair_rows(
    output_root: Path,
    manifest_rows: list[dict[str, str]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_dataset = load_per_pair(output_root / "per_pair.csv")
    for row in new_rows:
        by_dataset[row["dataset"]] = row

    ordered = []
    for row in manifest_rows:
        dataset = row["dataset"]
        if dataset not in by_dataset:
            continue
        merged = {key: by_dataset[dataset].get(key, "") for key in PER_PAIR_FIELDS}
        merged["dataset"] = dataset
        if not merged.get("category"):
            merged["category"] = row.get("category", "")
        ordered.append(merged)
    return ordered


def is_ok(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and row.get("status") == "ok"
        and math.isfinite(parse_float(row.get("mdr_rmse")))
        and math.isfinite(parse_float(row.get("niqe")))
    )


def write_comparison(output_root: Path, candidate_rows: list[dict[str, Any]], depth_gsp_root: Path) -> None:
    baseline = load_per_pair(depth_gsp_root / "per_pair.csv")
    candidate = {row["dataset"]: row for row in candidate_rows}
    names = sorted(set(baseline) | set(candidate))
    rows = []
    common_ok = []
    for name in names:
        base = baseline.get(name)
        cand = candidate.get(name)
        base_ok = is_ok(base)
        cand_ok = is_ok(cand)
        if base_ok and cand_ok:
            common_ok.append(name)
        base_mdr = parse_float(base.get("mdr_rmse")) if base else math.nan
        cand_mdr = parse_float(cand.get("mdr_rmse")) if cand else math.nan
        base_niqe = parse_float(base.get("niqe")) if base else math.nan
        cand_niqe = parse_float(cand.get("niqe")) if cand else math.nan
        rows.append(
            {
                "dataset": name,
                "category": (cand or base or {}).get("category", ""),
                "baseline_status": base.get("status", "missing") if base else "missing",
                "candidate_status": cand.get("status", "missing") if cand else "missing",
                "baseline_mdr": fmt_float(base_mdr),
                "candidate_mdr": fmt_float(cand_mdr),
                "mdr_delta": fmt_float(cand_mdr - base_mdr if base_ok and cand_ok else math.nan),
                "baseline_niqe": fmt_float(base_niqe),
                "candidate_niqe": fmt_float(cand_niqe),
                "niqe_delta": fmt_float(cand_niqe - base_niqe if base_ok and cand_ok else math.nan),
                "candidate_result_image": cand.get("result_image", "") if cand else "",
                "baseline_result_image": base.get("result_image", "") if base else "",
            }
        )
    write_csv(
        output_root / "method_pair_comparison.csv",
        rows,
        [
            "dataset",
            "category",
            "baseline_status",
            "candidate_status",
            "baseline_mdr",
            "candidate_mdr",
            "mdr_delta",
            "baseline_niqe",
            "candidate_niqe",
            "niqe_delta",
            "candidate_result_image",
            "baseline_result_image",
        ],
    )

    cand_mdr = finite_mean([parse_float(candidate[name].get("mdr_rmse")) for name in common_ok])
    base_mdr = finite_mean([parse_float(baseline[name].get("mdr_rmse")) for name in common_ok])
    cand_niqe = finite_mean([parse_float(candidate[name].get("niqe")) for name in common_ok])
    base_niqe = finite_mean([parse_float(baseline[name].get("niqe")) for name in common_ok])
    lines = [
        "# MGDH vs Depth-GSP-v5",
        "",
        f"- Method variant: {METHOD_VARIANT}",
        f"- Total datasets: {len(names)}",
        f"- Common successful datasets: {len(common_ok)}",
        f"- Common mean MDR: MGDH {fmt_float(cand_mdr)} vs Depth-GSP-v5 {fmt_float(base_mdr)}",
        f"- Common mean NIQE: MGDH {fmt_float(cand_niqe)} vs Depth-GSP-v5 {fmt_float(base_niqe)}",
        "",
        "Metric note: MDR is computed from the MGDH predicted 8x8 mesh on resized 512x512 pairs with OpenCV feature matches. NIQE is computed on `panorama.png`, which is rendered by the classical feature-homography canvas renderer.",
        "",
    ]
    (output_root / "method_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if is_ok(row)]
    mean_mdr = finite_mean([parse_float(row.get("mdr_rmse")) for row in ok_rows])
    mean_niqe = finite_mean([parse_float(row.get("niqe")) for row in ok_rows])
    lines = [
        "# MGDH StitchBench General Report",
        "",
        f"- Method variant: {METHOD_VARIANT}",
        f"- Total rows: {len(rows)}",
        f"- OK rows: {len(ok_rows)}",
        f"- Mean MDR: {fmt_float(mean_mdr)}",
        f"- Mean NIQE: {fmt_float(mean_niqe)}",
        "",
        "Result image: `<scene>/panorama.png`.",
        "MDR uses the MGDH predicted 8x8 mesh at 512x512 resolution; NIQE uses pyiqa on the saved panorama rendered by the classical canvas renderer.",
        "",
    ]
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_and_write(
    *,
    manifest: Path,
    mgdh_root: Path,
    output_root: Path,
    depth_gsp_root: Path,
    device: str,
    scenes: list[str] | None = None,
    limit: int = 0,
    skip_mdr: bool = False,
    skip_niqe: bool = False,
    skip_compare: bool = False,
) -> list[dict[str, Any]]:
    manifest_rows = read_manifest(manifest)
    rows = select_rows(manifest_rows, scenes=scenes, limit=limit)
    output_root.mkdir(parents=True, exist_ok=True)
    result_rows = evaluate_rows(rows, mgdh_root=mgdh_root, device=device, skip_mdr=skip_mdr, skip_niqe=skip_niqe)
    if scenes or limit > 0:
        result_rows = merge_per_pair_rows(output_root, manifest_rows, result_rows)
    write_csv(output_root / "per_pair.csv", result_rows, PER_PAIR_FIELDS)
    write_by_category(output_root, result_rows)
    write_report(output_root, result_rows)
    if not skip_compare:
        write_comparison(output_root, result_rows, depth_gsp_root)
    return result_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--mgdh-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--depth-gsp-root", default=DEFAULT_DEPTH_GSP_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-mdr", action="store_true")
    parser.add_argument("--skip-niqe", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = evaluate_and_write(
        manifest=Path(args.manifest),
        mgdh_root=Path(args.mgdh_root),
        output_root=Path(args.output_root),
        depth_gsp_root=Path(args.depth_gsp_root),
        device=args.device,
        scenes=args.scene,
        limit=args.limit,
        skip_mdr=args.skip_mdr,
        skip_niqe=args.skip_niqe,
        skip_compare=args.skip_compare,
    )
    ok_count = sum(1 for row in rows if is_ok(row))
    print(f"Wrote MGDH MDR/NIQE evaluation to {args.output_root}; ok={ok_count}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
