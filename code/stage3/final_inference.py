#!/usr/bin/env python3
"""Single-GPU final Stage3 inference pipeline.

The command accepts one joint PGD1+PGD2 checkpoint and performs, in order:

    noisy.npy
      -> overlapping-patch PGD1/PGD2 forward
      -> pgd1.npy / pgd2_raw.npy / raw denoised.npy
      -> local-alpha
      -> learned multi-scale Surface Gate
      -> tangent-plane Coverage Repair
      -> final denoised.npy and an optional submission zip

The neural network implementation is deliberately taken from this repository's
stage3 source tree.  This file only adds the directory driver and inlines the
three CPU post-processing passes so that the whole inference recipe has one
entry point.  Defaults are the production parameters used for the stable
81.73 B-board submission.

This script is intentionally single-GPU.  A 4090 is selected with --gpu-id;
no MPI launcher is used and no model parameter is changed during inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Runtime setup.  Jittor must see CUDA/compiler settings before it is imported.
# ---------------------------------------------------------------------------


def configure_environment(gpu_id: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OBJ_MESH_CACHE_MB", "512")
    os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
    compiler = Path(sys.prefix) / "bin" / "g++"
    if compiler.is_file():
        os.environ["cc_path"] = str(compiler)
    else:
        system_compiler = shutil.which("g++")
        if system_compiler:
            os.environ["cc_path"] = system_compiler
    os.environ.pop("DISPLAY", None)
    os.environ.pop("XAUTHORITY", None)


JT = None
OmegaConf = None
get_model = None
patch_based_denoise = None
to_normal_frame = None
from_normal_frame = None


def load_stage3_runtime(code_root: Path):
    """Import Jittor and the repository model only after CUDA setup."""
    global JT, OmegaConf, get_model, patch_based_denoise
    global to_normal_frame, from_normal_frame
    code_root = code_root.resolve()
    stage3 = code_root / "stage3"
    if not (stage3 / "src").is_dir():
        raise FileNotFoundError(f"stage3 source tree not found below {code_root}")
    os.chdir(code_root)
    if str(stage3) not in sys.path:
        sys.path.insert(0, str(stage3))
    import jittor as jt

    jt.flags.use_cuda = 1
    from omegaconf import OmegaConf as _OmegaConf
    from src.model.ops import (
        from_normal_frame as _from_normal_frame,
        patch_based_denoise as _patch_based_denoise,
        to_normal_frame as _to_normal_frame,
    )
    from src.model.parse import get_model as _get_model

    JT = jt
    OmegaConf = _OmegaConf
    get_model = _get_model
    patch_based_denoise = _patch_based_denoise
    to_normal_frame = _to_normal_frame
    from_normal_frame = _from_normal_frame


# ---------------------------------------------------------------------------
# Small NumPy inference-time components.  These are intentionally in this file
# rather than hidden behind a chain of post-processing shell commands.
# ---------------------------------------------------------------------------


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def knn_numpy(points: np.ndarray, k: int):
    from scipy.spatial import cKDTree

    k = min(max(1, int(k)), len(points))
    return cKDTree(points).query(points, k=k, workers=-1)


def refine_feature_preserving_normals(
    points: np.ndarray,
    *,
    k: int,
    strength: float = 1.0,
    max_step_ratio: float = 0.05,
    spatial_sigma: float = 0.70,
    normal_sigma: float = 0.20,
    plane_sigma: float = 0.20,
    gate_low: float = 0.15,
    gate_high: float = 0.65,
) -> np.ndarray:
    """One conservative, feature-gated, normal-only surface update."""
    points = np.asarray(points, dtype=np.float64)
    k = min(max(12, int(k)), len(points))
    distances, indices = knn_numpy(points, k)
    neighborhoods = points[indices]
    centers = neighborhoods.mean(axis=1, keepdims=True)
    centered = neighborhoods - centers
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / float(len(indices[0]))
    eigenvalues, bases = np.linalg.eigh(covariance)
    normals = bases[:, :, 0]
    planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(eigenvalues[:, 2], 1e-15)
    gates = smoothstep((planarity - gate_low) / max(gate_high - gate_low, 1e-12))
    radii = np.maximum(distances[:, -1], 1e-12)

    neighbor_normals = normals[indices]
    center_normals = normals
    alignment = np.einsum("nkj,nj->nk", neighbor_normals, center_normals, optimize=True)
    oriented_normals = neighbor_normals * np.where(alignment >= 0.0, 1.0, -1.0)[:, :, None]
    normal_difference = 1.0 - np.abs(alignment)
    spatial_weight = np.exp(-0.5 * (distances / (spatial_sigma * radii[:, None])) ** 2)
    normal_weight = np.exp(-0.5 * (normal_difference / normal_sigma) ** 2)
    residual = np.einsum("nkj,nkj->nk", points[:, None, :] - neighborhoods, oriented_normals, optimize=True)
    plane_weight = np.exp(-0.5 * (residual / (plane_sigma * radii[:, None])) ** 2)
    weights = spatial_weight * normal_weight * plane_weight
    correction = -(weights[:, :, None] * residual[:, :, None] * oriented_normals).sum(axis=1)
    correction /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-15)
    normal_scalar = np.einsum("nj,nj->n", correction, center_normals, optimize=True)[:, None]
    normal_step = normal_scalar * center_normals * float(strength) * gates[:, None]
    step_norm = np.linalg.norm(normal_step, axis=1)
    limiter = np.minimum(1.0, (max_step_ratio * radii) / np.maximum(step_norm, 1e-15))
    return (points + normal_step * limiter[:, None]).astype(np.float32)


def kernel_basis(points: np.ndarray, ks=(8, 12, 16, 24, 32)) -> np.ndarray:
    bases = [
        refine_feature_preserving_normals(points, k=int(k), strength=1.0, max_step_ratio=0.05).astype(np.float64) - points.astype(np.float64)
        for k in ks
    ]
    return np.stack(bases, axis=2)


def geometry_design(points: np.ndarray, basis: np.ndarray, weights: np.ndarray):
    """Exact feature construction used by the trained Surface Gate."""
    distances, indices = knn_numpy(points, 32)

    def eigenvalues_for(index_slice):
        neighborhoods = points[index_slice]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / float(index_slice.shape[1])
        return np.linalg.eigvalsh(covariance)

    split = min(16, indices.shape[1])
    eig16 = eigenvalues_for(indices[:, :split])
    eig32 = eigenvalues_for(indices)

    def descriptors(eigenvalues):
        largest = np.maximum(eigenvalues[:, 2], 1e-15)
        total = np.maximum(eigenvalues.sum(axis=1), 1e-15)
        planarity = np.clip((eigenvalues[:, 1] - eigenvalues[:, 0]) / largest, 0.0, 1.0)
        linearity = np.clip((eigenvalues[:, 2] - eigenvalues[:, 1]) / largest, 0.0, 1.0)
        variation = np.clip(eigenvalues[:, 0] / total, 0.0, 1.0)
        return planarity, linearity, variation

    p16, l16, s16 = descriptors(eig16)
    p32, l32, s32 = descriptors(eig32)
    correction = np.einsum("nck,k->nc", basis, np.asarray(weights), optimize=True)
    correction_norm = np.linalg.norm(correction, axis=1)
    radius16 = np.maximum(distances[:, split - 1], 1e-12)
    radius32 = np.maximum(distances[:, -1], 1e-12)
    relative_step = np.clip(correction_norm / radius16 * 20.0, 0.0, 2.0)
    fine = basis[:, :, 0]
    coarse = basis[:, :, -1]
    agreement = np.einsum("ni,ni->n", fine, coarse, optimize=True) / np.maximum(
        np.linalg.norm(fine, axis=1) * np.linalg.norm(coarse, axis=1), 1e-15
    )
    agreement = np.clip(agreement, -1.0, 1.0)
    density_ratio = np.clip(radius16 / radius32, 0.0, 1.0)
    raw = np.column_stack([p16, l16, s16 * 3.0, p32, l32, s32 * 3.0, relative_step, agreement, density_ratio])
    design = np.column_stack([
        np.ones(len(points)), raw, raw * raw,
        p16 * agreement, p32 * agreement, relative_step * agreement, s16 * density_ratio,
    ])
    return correction, design


def pgd_confidence_features(points, correction, pgd1, pgd2_raw, k=16):
    distances, indices = knn_numpy(points, k)
    radius = np.maximum(distances[:, -1], 1e-12)
    displacement = np.asarray(pgd2_raw, dtype=np.float64) - np.asarray(pgd1, dtype=np.float64)
    neighbor = displacement[indices]
    consensus = neighbor.mean(axis=1)
    disagreement = displacement - consensus
    local_std = np.sqrt(np.mean(np.sum((neighbor - consensus[:, None]) ** 2, axis=2), axis=1))

    def norm(x):
        return np.linalg.norm(x, axis=1)

    disp_norm = norm(displacement)
    consensus_norm = norm(consensus)
    disagreement_norm = norm(disagreement)
    correction_norm = norm(correction)

    def cosine(left, right, left_norm, right_norm):
        return np.clip(np.einsum("ni,ni->n", left, right, optimize=True) / np.maximum(left_norm * right_norm, 1e-15), -1.0, 1.0)

    return np.column_stack([
        np.clip(disp_norm / radius, 0.0, 2.0),
        np.clip(consensus_norm / radius, 0.0, 2.0),
        np.clip(disagreement_norm / radius, 0.0, 2.0),
        np.clip(local_std / radius, 0.0, 2.0),
        cosine(displacement, consensus, disp_norm, consensus_norm),
        cosine(displacement, correction, disp_norm, correction_norm),
        np.clip(norm(points - pgd2_raw) / radius, 0.0, 1.0),
        np.clip(norm(points - pgd1) / radius, 0.0, 2.0),
    ]).astype(np.float64)


class GateModel:
    """Small NumPy MLP loader matching surface_gate_mlp.GateMLP."""

    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        layers = int(data["layers"][0])
        self.weights = [data[f"weight_{i}"].astype(np.float32) for i in range(layers)]
        self.biases = [data[f"bias_{i}"].astype(np.float32) for i in range(layers)]
        self.mean = data["feature_mean"].astype(np.float32)
        self.std = data["feature_std"].astype(np.float32)
        self.max_gate = float(data["max_gate"][0])

    def forward(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features, dtype=np.float32) - self.mean) / np.maximum(self.std, 1e-6)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ weight + bias)
        logits = x @ self.weights[-1] + self.biases[-1]
        logits = np.clip(logits, -20.0, 20.0)
        return (self.max_gate / (1.0 + np.exp(-logits)))[:, 0]


def refine_coverage(points: np.ndarray, *, strength=0.90, max_step_ratio=0.01, k=16) -> np.ndarray:
    """Final tangent-only redistribution; no trainable model is used."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    k = min(max(12, int(k)), len(points))
    distances, indices = cKDTree(points).query(points, k=k, workers=-1)
    neighborhoods = points[indices]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / float(k)
    eigenvalues, bases = np.linalg.eigh(covariance)
    normals = bases[:, :, 0]
    planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(eigenvalues[:, 2], 1e-15)
    gate = smoothstep((planarity - 0.15) / 0.50)
    radius = np.maximum(distances[:, -1], 1e-12)
    offsets = neighborhoods[:, 1:] - points[:, None, :]
    neighbor_normals = normals[indices[:, 1:]]
    alignment = np.abs(np.einsum("nkj,nj->nk", neighbor_normals, normals, optimize=True))
    normalized_distance = distances[:, 1:] / radius[:, None]
    weights = np.exp(-0.5 * (normalized_distance / 0.55) ** 2)
    weights *= np.exp(-0.5 * ((1.0 - alignment) / 0.20) ** 2)
    weights /= np.maximum(normalized_distance, 0.08)
    local_centroid = np.einsum("nk,nkj->nj", weights, offsets, optimize=True) / np.maximum(weights.sum(axis=1, keepdims=True), 1e-15)
    normal_component = np.einsum("nj,nj->n", local_centroid, normals, optimize=True)[:, None] * normals
    tangent_centroid = local_centroid - normal_component
    step = -float(strength) * gate[:, None] * tangent_centroid
    step_norm = np.linalg.norm(step, axis=1)
    step *= np.minimum(1.0, (float(max_step_ratio) * radius) / np.maximum(step_norm, 1e-15))[:, None]
    return (points + step).astype(np.float32)


# ---------------------------------------------------------------------------
# PGD1/PGD2 forward, copied in behaviour from stage3/run.py's production path.
# ---------------------------------------------------------------------------


def seed_all(seed: int) -> None:
    JT.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_stage_pair(checkpoint: Path):
    params = JT.load(str(checkpoint))
    stage1, stage2 = {}, {}
    for key, value in params.items():
        key = str(key)
        if key.startswith("pgd1."):
            stage1[key[5:]] = value
        elif key.startswith("pgd2."):
            stage2[key[5:]] = value
    if not stage1 or not stage2:
        raise ValueError("checkpoint must contain pgd1.* and pgd2.* parameters")
    return stage1, stage2


class Predictor:
    def __init__(self, stage1, stage2, args):
        self.stage1 = stage1
        self.stage2 = stage2
        self.patch_size = args.patch_size
        self.seed_k = args.patch_seed_k
        self.seed_k_alpha = args.patch_seed_k_alpha
        self.patch_merge = args.patch_merge
        self.patch_merge_beta = args.patch_merge_beta
        self.patch_adaptive_alpha_max = args.patch_adaptive_alpha_max
        self.patch_adaptive_tau = args.patch_adaptive_tau
        self.normal_align_stage2 = True
        self.niters = int(stage1.niters)

    def denoise_langevin_dynamics(self, patches, return_intermediates=False, **_kwargs):
        with JT.no_grad():
            disp1, _ = self.stage1.feature_nets(patches, calculate_commitment_losses=False)
            x1 = patches + disp1
            x1_local, frames = to_normal_frame(x1, sign_mode="stable_dominant")
            disp2, _ = self.stage2.feature_nets(x1_local, calculate_commitment_losses=False)
            x2 = from_normal_frame(x1_local + disp2, frames)
            return (x2, x1) if return_intermediates else x2

    def predict(self, points: np.ndarray):
        current = JT.array(np.asarray(points, dtype=np.float32))
        intermediate = None
        for _ in range(self.niters):
            current, intermediate = patch_based_denoise(
                model=self,
                pcl_noisy=current,
                patch_size=self.patch_size,
                seed_k=self.seed_k,
                seed_k_alpha=self.seed_k_alpha,
                merge_mode=self.patch_merge,
                merge_beta=self.patch_merge_beta,
                merge_adaptive_alpha_max=self.patch_adaptive_alpha_max,
                merge_adaptive_tau=self.patch_adaptive_tau,
                return_intermediates=True,
            )
        return current.numpy().astype(np.float32), intermediate.numpy().astype(np.float32)


def atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    tmp.replace(path)


def entries_from_list(path: Path) -> list[str]:
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"empty test list: {path}")
    if len(set(entries)) != len(entries):
        raise ValueError(f"duplicate entries in test list: {path}")
    return entries


def expected_paths(input_root: Path, entries: Sequence[str]) -> list[Path]:
    return [input_root / entry / "noisy.npy" for entry in entries]


def forward_one(task):
    entry, input_root, intermediate_root, checkpoint, args_dict = task
    source = Path(input_root) / entry / "noisy.npy"
    target_dir = Path(intermediate_root) / entry
    if not source.is_file():
        raise FileNotFoundError(source)
    if not args_dict["overwrite"] and all((target_dir / name).is_file() for name in ("denoised.npy", "pgd1.npy", "pgd2_raw.npy")):
        return entry, "skipped"

    # Child workers are not used for forward; this function is only retained
    # for a clear manifest and is not submitted to ProcessPoolExecutor.
    raise RuntimeError("forward_one must be run in the main GPU process")


def run_forward(args, entries: Sequence[str], input_root: Path, work_root: Path, code_root: Path):
    configure_environment(args.gpu_id)
    load_stage3_runtime(code_root)
    seed_all(args.seed)
    model_config = OmegaConf.to_container(OmegaConf.load(str(args.model_config)))
    transform_config = OmegaConf.to_container(OmegaConf.load(str(args.transform_config)))
    stage1 = get_model(model_config=model_config, transform_config=transform_config)
    stage2 = get_model(model_config=model_config, transform_config=transform_config)
    params1, params2 = split_stage_pair(args.checkpoint)
    stage1.load_parameters(params1)
    stage2.load_parameters(params2)
    stage1.set_predict(True)
    stage2.set_predict(True)
    stage1.eval()
    stage2.eval()
    predictor = Predictor(stage1, stage2, args)

    intermediate_root = work_root / "intermediates"
    rows = []
    started = time.time()
    for index, entry in enumerate(entries, 1):
        target_dir = intermediate_root / entry
        if not args.overwrite and all((target_dir / name).is_file() for name in ("denoised.npy", "pgd1.npy", "pgd2_raw.npy")):
            rows.append((entry, "skipped"))
            print(f"[forward] {index}/{len(entries)} skipped {entry}", flush=True)
            continue
        points = np.load(input_root / entry / "noisy.npy").astype(np.float32)
        if points.shape != (50000, 3) or not np.isfinite(points).all():
            raise ValueError(f"expected finite (50000, 3) noisy points: {entry}, got {points.shape}")
        denoised, pgd1 = predictor.predict(points)
        atomic_save(target_dir / "denoised.npy", denoised)
        atomic_save(target_dir / "pgd1.npy", pgd1)
        # The historical pipeline names the already-fused PGD2 output raw.
        atomic_save(target_dir / "pgd2_raw.npy", denoised)
        rows.append((entry, "written"))
        if index == 1 or index % 10 == 0 or index == len(entries):
            elapsed = time.time() - started
            print(f"[forward] {index}/{len(entries)} elapsed={elapsed/60:.2f} min", flush=True)
    return rows


def local_alpha_one(source: Path, destination: Path, args) -> tuple[str, str]:
    pgd1 = np.load(source / "pgd1.npy").astype(np.float64)
    raw = np.load(source / "pgd2_raw.npy").astype(np.float64)
    displacement = raw - pgd1
    distances, indices = knn_numpy(pgd1, args.local_k)
    radius = np.maximum(distances[:, -1], 1e-12)
    consensus = displacement[indices].mean(axis=1)
    ratio = np.linalg.norm(displacement - consensus, axis=1) / radius
    risk = smoothstep((ratio - args.local_band_low) / (args.local_band_high - args.local_band_low))
    confidence = 1.0 - risk
    alpha = args.local_base + args.local_delta * (2.0 * confidence - 1.0)
    alpha = np.clip(alpha, 0.0, 1.8)
    atomic_save(destination / "denoised.npy", pgd1 + alpha[:, None] * displacement)
    return str(source), "written"


def run_local_alpha(args, entries: Sequence[str], work_root: Path):
    source_root = work_root / "intermediates"
    output_root = work_root / "local_alpha"
    tasks = []
    for entry in entries:
        source = source_root / entry
        destination = output_root / entry
        if not args.overwrite and (destination / "denoised.npy").is_file():
            continue
        tasks.append((source, destination, args))
    if not tasks:
        print("[local-alpha] all files already exist", flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.post_workers) as executor:
        futures = [executor.submit(local_alpha_one, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index == 1 or index % 10 == 0 or index == len(futures):
                print(f"[local-alpha] {index}/{len(futures)}", flush=True)


def gate_one(source: Path, destination: Path, intermediate: Path, model_paths: Sequence[Path], args):
    points = np.load(source / "denoised.npy").astype(np.float32)
    pgd1 = np.load(intermediate / "pgd1.npy").astype(np.float32)
    raw = np.load(intermediate / "pgd2_raw.npy").astype(np.float32)
    basis = kernel_basis(points, args.surface_ks)
    correction, design = geometry_design(points, basis, np.asarray(args.surface_weights, dtype=np.float64))
    design = np.column_stack([design, pgd_confidence_features(points, correction, pgd1, raw)])
    models = [GateModel(path) for path in model_paths]
    expected_features = models[0].mean.shape[0]
    if design.shape[1] - 1 != expected_features:
        raise ValueError(f"gate feature mismatch: design={design.shape[1]-1} model={expected_features}")
    gate = args.gate_scale * np.mean([model.forward(design[:, 1:]) for model in models], axis=0)
    output = points.astype(np.float64) + gate[:, None] * correction
    output = refine_coverage(output.astype(np.float32), strength=args.coverage_strength, max_step_ratio=args.coverage_max_step_ratio, k=args.coverage_k)
    atomic_save(destination / "denoised.npy", output)
    return str(source), "written"


def run_surface_and_coverage(args, entries: Sequence[str], work_root: Path, model_dir: Path):
    source_root = work_root / "local_alpha"
    intermediate_root = work_root / "intermediates"
    output_root = work_root / "final"
    model_paths = sorted(model_dir.glob("gate_fold*.npz"))
    if len(model_paths) != 5:
        raise FileNotFoundError(f"expected five gate_fold*.npz files in {model_dir}, found {len(model_paths)}")
    tasks = []
    for entry in entries:
        source = source_root / entry
        destination = output_root / entry
        if not args.overwrite and (destination / "denoised.npy").is_file():
            continue
        tasks.append((source, destination, intermediate_root / entry, model_paths, args))
    if not tasks:
        print("[surface+coverage] all files already exist", flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.post_workers) as executor:
        futures = [executor.submit(gate_one, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index == 1 or index % 10 == 0 or index == len(futures):
                print(f"[surface+coverage] {index}/{len(futures)}", flush=True)


def validate_predictions(result_root: Path, entries: Sequence[str]):
    expected = {result_root / entry / "denoised.npy" for entry in entries}
    actual = set(result_root.rglob("denoised.npy"))
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(f"final set mismatch: missing={len(missing)} extra={len(extra)}")
    for path in expected:
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (50000, 3) or not np.isfinite(arr).all():
            raise ValueError(f"invalid prediction: {path} shape={arr.shape}")
    return len(expected)


def package_submission(result_root: Path, entries: Sequence[str], output_zip: Path):
    validate_predictions(result_root, entries)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            archive.write(result_root / entry / "denoised.npy", arcname=str(Path(entry) / "denoised.npy"))
    with zipfile.ZipFile(output_zip) as archive:
        names = archive.namelist()
        if len(names) != len(entries):
            raise RuntimeError(f"archive entry count mismatch: {len(names)} != {len(entries)}")
    print(f"[package] validated={len(entries)} archive={output_zip} size_mb={output_zip.stat().st_size/1024/1024:.2f}", flush=True)


def write_manifest(args, entries: Sequence[str], work_root: Path, model_dir: Path):
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "entries": len(entries),
        "gpu_id": args.gpu_id,
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": {
            "patch_size": args.patch_size,
            "patch_seed_k": args.patch_seed_k,
            "patch_seed_k_alpha": args.patch_seed_k_alpha,
            "patch_merge": args.patch_merge,
            "patch_merge_beta": args.patch_merge_beta,
            "patch_adaptive_alpha_max": args.patch_adaptive_alpha_max,
            "patch_adaptive_tau": args.patch_adaptive_tau,
            "normal_align_stage2": True,
            "normal_frame_sign_mode": "stable_dominant",
            "tta_passes": 1,
            "local_base": args.local_base,
            "local_delta": args.local_delta,
            "local_band": [args.local_band_low, args.local_band_high],
            "gate_scale": args.gate_scale,
            "coverage_strength": args.coverage_strength,
            "coverage_max_step_ratio": args.coverage_max_step_ratio,
        },
        "gate_models": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(model_dir.glob("gate_fold*.npz"))
        },
    }
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args():
    here = Path(__file__).resolve().parent
    # In the main repository this file lives inside ``code/stage3`` and should
    # import the sibling ``src`` package through the parent ``code`` directory.
    # The fallback preserves the old self-contained inference_repro layout.
    code_root = here if (here / "stage3").is_dir() else here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True, help="List of relative sample directories, one per line.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path)
    parser.add_argument("--mode", choices=("all", "forward", "postprocess", "package"), default="all")
    parser.add_argument("--code-root", type=Path, default=code_root)
    parser.add_argument("--model-config", type=Path, default=code_root / "stage3/configs/model/core.yaml")
    parser.add_argument("--transform-config", type=Path, default=code_root / "stage3/configs/transform/ops.yaml")
    parser.add_argument("--models-dir", type=Path, default=here / "models/surface_gate")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--seed", type=int, default=46105)
    parser.add_argument("--post-workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")

    # Production forward parameters.
    parser.add_argument("--patch-size", type=int, default=2600)
    parser.add_argument("--patch-seed-k", type=int, default=52)
    parser.add_argument("--patch-seed-k-alpha", type=int, default=10)
    parser.add_argument("--patch-merge", choices=("hard", "weighted", "adaptive"), default="adaptive")
    parser.add_argument("--patch-merge-beta", type=float, default=8.0)
    parser.add_argument("--patch-adaptive-alpha-max", type=float, default=1.0)
    parser.add_argument("--patch-adaptive-tau", type=float, default=0.008)

    # Production local-alpha parameters.
    parser.add_argument("--local-k", type=int, default=16)
    parser.add_argument("--local-base", type=float, default=1.20)
    parser.add_argument("--local-delta", type=float, default=0.07)
    parser.add_argument("--local-band-low", type=float, default=0.05)
    parser.add_argument("--local-band-high", type=float, default=0.25)

    # Production Surface Gate and Coverage Repair parameters.
    parser.add_argument("--surface-ks", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--surface-weights", type=float, nargs="+", default=[0.2, 0.2, 0.2, 0.2, 0.2])
    parser.add_argument("--gate-scale", type=float, default=0.575)
    parser.add_argument("--coverage-strength", type=float, default=0.90)
    parser.add_argument("--coverage-max-step-ratio", type=float, default=0.01)
    parser.add_argument("--coverage-k", type=int, default=16)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.input_root = args.input_root.resolve()
    args.test_list = args.test_list.resolve()
    args.work_dir = args.work_dir.resolve()
    args.code_root = args.code_root.resolve()
    args.model_config = args.model_config.resolve()
    args.transform_config = args.transform_config.resolve()
    args.models_dir = args.models_dir.resolve()
    if args.output_zip is None:
        args.output_zip = args.work_dir / "result.zip"
    else:
        args.output_zip = args.output_zip.resolve()
    if len(args.surface_ks) != len(args.surface_weights):
        parser.error("--surface-ks and --surface-weights must have equal lengths")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not args.input_root.is_dir():
        parser.error(f"input root not found: {args.input_root}")
    return args


def main() -> int:
    args = parse_args()
    entries = entries_from_list(args.test_list)
    files = expected_paths(args.input_root, entries)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing noisy.npy files, first={missing[0]}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, entries, args.work_dir, args.models_dir)

    if args.mode in ("all", "forward"):
        run_forward(args, entries, args.input_root, args.work_dir, args.code_root)
        if args.mode == "forward":
            return 0
    if args.mode in ("all", "postprocess"):
        run_local_alpha(args, entries, args.work_dir)
        run_surface_and_coverage(args, entries, args.work_dir, args.models_dir)
        if args.mode == "postprocess":
            return 0
    if args.mode in ("all", "package"):
        package_submission(args.work_dir / "final", entries, args.output_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
