#!/usr/bin/env python3
"""Compare two denoised prediction directories.

The script matches ``denoised.npy`` files by relative path and reports
pointwise distance statistics.  It can also compute symmetric nearest-neighbor
distances as a slower sanity check when point order may have changed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def read_entries(path: Path) -> list[str]:
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"empty test list: {path}")
    return entries


def discover_prediction_paths(root: Path, filename: str) -> dict[str, Path]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): path
        for path in root.rglob(filename)
        if path.is_file()
    }


def expected_prediction_paths(root: Path, entries: Sequence[str], filename: str) -> dict[str, Path]:
    return {
        str(Path(entry) / filename).replace("\\", "/"): root / entry / filename
        for entry in entries
    }


def load_cloud(path: Path, *, expected_points: int | None) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected (N, 3) array at {path}, got {arr.shape}")
    if expected_points is not None and arr.shape[0] != expected_points:
        raise ValueError(f"expected ({expected_points}, 3) at {path}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite values in {path}")
    return arr.astype(np.float64, copy=False)


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def pointwise_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    diff = left - right
    distances = np.linalg.norm(diff, axis=1)
    abs_diff = np.abs(diff)
    return {
        "point_l2_mean": float(distances.mean()),
        "point_l2_median": float(np.median(distances)),
        "point_l2_p95": percentile(distances, 95),
        "point_l2_p99": percentile(distances, 99),
        "point_l2_max": float(distances.max()),
        "coord_abs_mean": float(abs_diff.mean()),
        "coord_abs_max": float(abs_diff.max()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def chamfer_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree

    left_to_right, _ = cKDTree(right).query(left, k=1, workers=-1)
    right_to_left, _ = cKDTree(left).query(right, k=1, workers=-1)
    symmetric = np.concatenate([left_to_right, right_to_left])
    return {
        "nn_l2_mean": float(symmetric.mean()),
        "nn_l2_p95": percentile(symmetric, 95),
        "nn_l2_p99": percentile(symmetric, 99),
        "nn_l2_max": float(symmetric.max()),
        "chamfer_l2_squared_mean": float(np.mean(left_to_right ** 2) + np.mean(right_to_left ** 2)),
    }


def summarize(rows: list[dict[str, float | str]]) -> dict[str, float | str | int]:
    numeric_keys = [key for key in rows[0] if key != "relative_path"]
    summary: dict[str, float | str | int] = {"samples": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean_over_samples"] = float(values.mean())
        summary[f"{key}_max_over_samples"] = float(values.max())
    worst = max(rows, key=lambda row: float(row["point_l2_max"]))
    summary["worst_point_l2_path"] = str(worst["relative_path"])
    summary["worst_point_l2_max"] = float(worst["point_l2_max"])
    return summary


def write_csv(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_dir", type=Path, help="Reference/submitted prediction root.")
    parser.add_argument("right_dir", type=Path, help="Reproduced prediction root.")
    parser.add_argument("--filename", default="denoised.npy")
    parser.add_argument("--test-list", type=Path, help="Optional list of sample directories relative to each root.")
    parser.add_argument("--expected-points", type=int, default=50000)
    parser.add_argument("--nearest-neighbor", action="store_true", help="Also compute symmetric nearest-neighbor distances.")
    parser.add_argument("--csv", type=Path, help="Optional per-sample CSV output.")
    parser.add_argument("--json", type=Path, help="Optional summary JSON output.")
    parser.add_argument("--atol-mean", type=float, help="Fail if global mean pointwise L2 is above this value.")
    parser.add_argument("--atol-max", type=float, help="Fail if any pointwise L2 is above this value.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_root = args.left_dir.resolve()
    right_root = args.right_dir.resolve()
    if args.test_list is None:
        left_paths = discover_prediction_paths(left_root, args.filename)
        right_paths = discover_prediction_paths(right_root, args.filename)
    else:
        entries = read_entries(args.test_list.resolve())
        left_paths = expected_prediction_paths(left_root, entries, args.filename)
        right_paths = expected_prediction_paths(right_root, entries, args.filename)

    missing_left = sorted(key for key, path in left_paths.items() if not path.is_file())
    missing_right = sorted(key for key, path in right_paths.items() if not path.is_file())
    extra_left = sorted(set(left_paths) - set(right_paths))
    extra_right = sorted(set(right_paths) - set(left_paths))
    if missing_left or missing_right or extra_left or extra_right:
        raise FileNotFoundError(
            "prediction sets do not match: "
            f"missing_left={len(missing_left)} missing_right={len(missing_right)} "
            f"extra_left={len(extra_left)} extra_right={len(extra_right)}"
        )

    rows: list[dict[str, float | str]] = []
    for index, key in enumerate(sorted(left_paths), 1):
        left = load_cloud(left_paths[key], expected_points=args.expected_points)
        right = load_cloud(right_paths[key], expected_points=args.expected_points)
        row: dict[str, float | str] = {"relative_path": key}
        row.update(pointwise_stats(left, right))
        if args.nearest_neighbor:
            row.update(chamfer_stats(left, right))
        rows.append(row)
        if index == 1 or index % 25 == 0 or index == len(left_paths):
            print(f"[compare] {index}/{len(left_paths)} {key}", flush=True)

    summary = summarize(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.csv is not None:
        write_csv(args.csv.resolve(), rows)
    if args.json is not None:
        args.json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.json.resolve().write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = False
    if args.atol_mean is not None and float(summary["point_l2_mean_mean_over_samples"]) > args.atol_mean:
        failed = True
    if args.atol_max is not None and float(summary["point_l2_max_max_over_samples"]) > args.atol_max:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
