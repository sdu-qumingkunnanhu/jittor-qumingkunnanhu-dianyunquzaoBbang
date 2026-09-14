import math
from typing import Dict, Optional, Tuple

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from src.utils.pointops import batch_gather, farthest_point_sampling


def _structure_aware_seed_points(
    pcl_noisy: jt.Var,
    uniform_seed_pnts: jt.Var,
    num_patches: int,
    structure_ratio: float,
    neighbor_k: int,
    candidate_multiplier: int,
    score_power: float,
) -> Tuple[jt.Var, int]:
    """Mix uniform FPS centers with spatially dispersed line-like centers.

    The total number of centers is unchanged. Local PCA is evaluated only for
    seed selection on a detached CPU copy, so PGD1/PGD2 are untouched.
    """
    structure_count = min(
        max(0, int(round(float(structure_ratio) * int(num_patches)))),
        max(0, int(num_patches) - 1),
    )
    if structure_count == 0:
        return uniform_seed_pnts, 0

    points = pcl_noisy.detach().numpy().astype(np.float32, copy=False)
    uniform = uniform_seed_pnts[0].detach().numpy().astype(np.float32, copy=False)
    point_count = int(points.shape[0])
    uniform_count = int(num_patches) - structure_count
    retained_uniform = uniform[:uniform_count]

    k = min(max(4, int(neighbor_k)), point_count)
    tree = cKDTree(points)
    linearity = np.empty((point_count,), dtype=np.float32)
    chunk_size = 4096
    for start in range(0, point_count, chunk_size):
        end = min(point_count, start + chunk_size)
        _, neighbor_indices = tree.query(points[start:end], k=k, workers=1)
        neighbors = points[neighbor_indices]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered)
        covariance /= np.float32(max(1, k - 1))
        eigenvalues = np.linalg.eigvalsh(covariance).astype(np.float32)
        linearity[start:end] = np.clip(
            (eigenvalues[:, 2] - eigenvalues[:, 1])
            / np.maximum(eigenvalues[:, 2], np.float32(1e-12)),
            0.0,
            1.0,
        )

    candidate_count = min(
        point_count,
        max(512, structure_count * max(1, int(candidate_multiplier))),
    )
    if candidate_count == point_count:
        candidate_indices = np.arange(point_count, dtype=np.int64)
    else:
        split = point_count - candidate_count
        candidate_indices = np.argpartition(linearity, split)[split:].astype(np.int64)
    candidates = points[candidate_indices]
    candidate_linearity = linearity[candidate_indices]

    min_distance2 = np.full((candidate_count,), np.inf, dtype=np.float32)
    for start in range(0, uniform_count, 256):
        reference = retained_uniform[start:start + 256]
        offset = candidates[:, None, :] - reference[None, :, :]
        distance2 = np.einsum("nki,nki->nk", offset, offset)
        min_distance2 = np.minimum(min_distance2, distance2.min(axis=1))

    structure_weight = np.float32(0.05) + np.float32(0.95) * np.power(
        candidate_linearity, np.float32(score_power)
    )
    selected = []
    available = np.ones((candidate_count,), dtype=bool)
    for _ in range(structure_count):
        priority = min_distance2 * structure_weight
        priority[~available] = -1.0
        local_index = int(np.argmax(priority))
        selected.append(int(candidate_indices[local_index]))
        available[local_index] = False
        offset = candidates - candidates[local_index]
        distance2 = np.einsum("ni,ni->n", offset, offset)
        min_distance2 = np.minimum(min_distance2, distance2)

    mixed = np.concatenate(
        [retained_uniform, points[np.asarray(selected, dtype=np.int64)]], axis=0
    ).astype(np.float32, copy=False)
    return jt.array(mixed[None, ...]), structure_count


def knn_points(x: jt.Var, y: jt.Var, k: int) -> Tuple[jt.Var, jt.Var, jt.Var]:
    """
    x: (B, P, C), y: (B, N, C)
    returns squared distances, indices, nearest neighbors.
    """
    if x.shape[-1] == 3 and y.shape[-1] == 3:
        dist, idx = jt.misc.knn(x, y, k)
    else:
        dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
        dist, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    nn = []
    for b in range(x.shape[0]):
        nn.append(y[b][idx[b]])
    return dist, idx, jt.stack(nn, dim=0)


def normalize_sphere(pc: jt.Var, radius: float = 1.0):
    p_max = jt.max(pc, dim=-2, keepdims=True)
    p_min = jt.min(pc, dim=-2, keepdims=True)
    center = (p_max + p_min) / 2.0
    pc = pc - center
    scale = jt.sqrt((pc ** 2).sum(dim=-1, keepdims=True))
    scale = jt.max(scale, dim=-2, keepdims=True) / radius
    return pc / (scale + 1e-8), center, scale


def chamfer_distance_unit_sphere(gen: jt.Var, ref: jt.Var):
    ref, center, scale = normalize_sphere(ref)
    gen = (gen - center) / (scale + 1e-8)
    dist_ab = ((gen.unsqueeze(2) - ref.unsqueeze(1)) ** 2).sum(-1)
    d1 = jt.min(dist_ab, dim=2)
    d2 = jt.min(dist_ab, dim=1)
    return d1.mean() + d2.mean()


def normal_alignment_frames(
    patches: jt.Var,
    sign_mode: str = "stable_dominant",
    pca_tangent_threshold: float = 0.15,
    tangent_phase_mode: str = "none",
    patch_indices=None,
) -> jt.Var:
    """Return deterministic PCA frames with the patch normal as local +z.

    The frame is intentionally treated as geometry metadata: PCA runs on a
    detached NumPy copy while gradients still flow through the subsequent
    matrix multiplications.  This follows the normal-alignment idea supplied
    with the updated Stage2/Stage3 code without changing the PGD backbone or
    its three-coordinate interface.
    """
    assert len(patches.shape) == 3 and patches.shape[-1] == 3, (
        "patches must be (B, N, 3)"
    )
    if sign_mode not in {"stable_dominant", "legacy_z", "hybrid_pca_tangent"}:
        raise ValueError(f"unsupported normal-frame sign mode: {sign_mode}")
    if not 0 <= pca_tangent_threshold <= 1:
        raise ValueError("pca_tangent_threshold must be in [0, 1]")
    if tangent_phase_mode not in {"none", "flip2", "cycle4"}:
        raise ValueError(f"unsupported tangent phase mode: {tangent_phase_mode}")

    # PCA is non-differentiable metadata.  Never call numpy() directly on a
    # Var that belongs to the trainable PGD graph: doing so finalizes Jittor's
    # lazy forward graph before MPI backward has consumed it.
    patches_np = patches.detach().numpy().astype("float32")
    frames = []
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if patch_indices is None:
        patch_indices_np = np.arange(len(patches_np), dtype=np.int64)
    else:
        patch_indices_np = np.asarray(patch_indices, dtype=np.int64).reshape(-1)
        if len(patch_indices_np) != len(patches_np):
            raise ValueError("patch_indices must match the patch batch size")

    for patch, patch_index in zip(patches_np, patch_indices_np):
        centered = patch - patch.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, patch.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0].astype(np.float32)
        normal /= np.linalg.norm(normal) + 1e-8

        # joint_461 was trained with the legacy +z-only convention.  Keep it
        # available for exact checkpoint compatibility; the dominant-axis
        # fallback is useful for newly trained models whose normals are nearly
        # perpendicular to z and would otherwise inherit an arbitrary sign.
        if sign_mode == "legacy_z":
            if normal[2] < 0:
                normal = -normal
        elif abs(float(normal[2])) > 1e-5:
            if normal[2] < 0:
                normal = -normal
        else:
            dominant = int(np.argmax(np.abs(normal)))
            if normal[dominant] < 0:
                normal = -normal

        largest = max(float(eigenvalues[2]), 1e-12)
        tangent_linearity = max(
            0.0, min(1.0, float((eigenvalues[2] - eigenvalues[1]) / largest))
        )
        if (
            sign_mode == "hybrid_pca_tangent"
            and tangent_linearity >= pca_tangent_threshold
        ):
            tangent1 = eigenvectors[:, 2].astype(np.float32)
            dominant = int(np.argmax(np.abs(tangent1)))
            if tangent1[dominant] < 0:
                tangent1 = -tangent1
        else:
            helper = z_axis if abs(float(np.dot(normal, z_axis))) < 0.9 else y_axis
            tangent1 = np.cross(helper, normal)
        tangent1 -= normal * float(np.dot(tangent1, normal))
        tangent1 /= np.linalg.norm(tangent1) + 1e-8
        tangent2 = np.cross(normal, tangent1)
        tangent2 /= np.linalg.norm(tangent2) + 1e-8
        if tangent_phase_mode != "none":
            divisor = 2 if tangent_phase_mode == "flip2" else 4
            phase = 2.0 * np.pi * (int(patch_index) % divisor) / divisor
            cosine = np.float32(np.cos(phase))
            sine = np.float32(np.sin(phase))
            base_tangent1 = tangent1.copy()
            base_tangent2 = tangent2.copy()
            tangent1 = cosine * base_tangent1 + sine * base_tangent2
            tangent2 = -sine * base_tangent1 + cosine * base_tangent2
        frames.append(np.stack([tangent1, tangent2, normal], axis=1))

    return jt.array(np.stack(frames, axis=0).astype("float32"))


def to_normal_frame(
    patches: jt.Var,
    frames: jt.Var = None,
    sign_mode: str = "stable_dominant",
    pca_tangent_threshold: float = 0.15,
    tangent_phase_mode: str = "none",
    patch_indices=None,
) -> Tuple[jt.Var, jt.Var]:
    """Rotate world xyz into a patch-local tangent/tangent/normal frame."""
    if frames is None:
        frames = normal_alignment_frames(
            patches,
            sign_mode=sign_mode,
            pca_tangent_threshold=pca_tangent_threshold,
            tangent_phase_mode=tangent_phase_mode,
            patch_indices=patch_indices,
        )
    return jt.matmul(patches, frames), frames


def from_normal_frame(patches: jt.Var, frames: jt.Var) -> jt.Var:
    """Rotate local tangent/tangent/normal coordinates back to world xyz."""
    return jt.matmul(patches, frames.transpose(0, 2, 1))


def patch_based_denoise(
    model,
    pcl_noisy: jt.Var,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=10,
    merge_mode="hard",
    merge_beta=4.0,
    merge_adaptive_alpha_max=0.7,
    merge_adaptive_tau=0.008,
    structure_seed_ratio=0.0,
    structure_neighbor_k=24,
    structure_candidate_multiplier=12,
    structure_score_power=2.0,
    structure_merge_boost=1.0,
    return_intermediates=False,
    diagnostics: Optional[Dict[str, np.ndarray]] = None,
):
    assert len(pcl_noisy.shape) == 2, "input point cloud must be (N, 3)"
    if merge_mode not in {"hard", "weighted", "adaptive"}:
        raise ValueError(f"unsupported patch merge mode: {merge_mode}")
    if merge_mode in {"weighted", "adaptive"} and merge_beta <= 0:
        raise ValueError("merge_beta must be greater than zero")
    if not 0 <= merge_adaptive_alpha_max <= 1:
        raise ValueError("merge_adaptive_alpha_max must be in [0, 1]")
    if merge_adaptive_tau <= 0:
        raise ValueError("merge_adaptive_tau must be greater than zero")
    if structure_merge_boost <= 0:
        raise ValueError("structure_merge_boost must be greater than zero")
    n_points, _ = pcl_noisy.shape
    num_patches = max(1, int(seed_k * n_points / patch_size))
    pcl_batch = pcl_noisy.unsqueeze(0)
    seed_pnts, _ = farthest_point_sampling(pcl_batch, num_patches)
    seed_pnts, structure_seed_count = _structure_aware_seed_points(
        pcl_noisy,
        seed_pnts,
        num_patches=num_patches,
        structure_ratio=structure_seed_ratio,
        neighbor_k=structure_neighbor_k,
        candidate_multiplier=structure_candidate_multiplier,
        score_power=structure_score_power,
    )
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_batch, patch_size)

    patches = patches[0]
    patch_dists = patch_dists[0]
    point_idxs = point_idxs[0]
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches_centered = patches - seed_expand

    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    best_patch = None
    if merge_mode in {"hard", "adaptive"}:
        all_dists = jt.ones((num_patches, n_points)) * 1e10
        for i in range(num_patches):
            all_dists[i][point_idxs[i]] = patch_dists[i]
        weights = jt.exp(-all_dists)
        best_patch, _ = jt.argmax(weights, dim=0)

    denoised_chunks = []
    intermediate_chunks = []
    step = max(1, int(math.ceil(n_points / (seed_k_alpha * patch_size))))
    i = 0
    while i < num_patches:
        end = min(num_patches, i + step)
        tangent_phase_mode = getattr(
            model, "normal_frame_tangent_phase_mode", "none"
        )
        phase_kwargs = {}
        if tangent_phase_mode != "none":
            phase_kwargs["patch_indices"] = np.arange(i, end, dtype=np.int64)
        if return_intermediates:
            denoised, intermediate = model.denoise_langevin_dynamics(
                patches_centered[i:end],
                return_intermediates=True,
                **phase_kwargs,
            )
            denoised_chunks.append(denoised)
            intermediate_chunks.append(intermediate)
        else:
            denoised_chunks.append(
                model.denoise_langevin_dynamics(
                    patches_centered[i:end], **phase_kwargs
                )
            )
        i = end
    patches_denoised = jt.concat(denoised_chunks, dim=0) + seed_expand

    def merge_patches(patches_prediction, record_diagnostics=False):
        input_np = pcl_noisy.numpy().astype("float32")
        idx_np = point_idxs.numpy().astype("int64")
        patches_np = patches_prediction.numpy().astype("float32")
        dist_np = patch_dists.numpy().astype("float32")
        flat_idx = idx_np.reshape(-1)
        flat_disp = patches_np.reshape(-1, 3) - input_np[flat_idx]

        hard_np = None
        best_np = None
        if merge_mode in {"hard", "adaptive"}:
            hard_np = input_np.copy()
            best_np = best_patch.numpy().astype("int64")
            for patch_id in range(num_patches):
                global_idx = idx_np[patch_id]
                selected = best_np[global_idx] == patch_id
                if not selected.any():
                    continue
                hard_np[global_idx[selected]] = patches_np[patch_id, selected]

        weighted_np = None
        if merge_mode in {"weighted", "adaptive"}:
            # Each global point occurs in several overlapping patches.  Average
            # predicted displacements with greater weight on patch-center views.
            weighted_np = input_np.copy()
            weight_np = np.exp(-np.float32(merge_beta) * dist_np).astype("float32")
            if structure_seed_count > 0 and structure_merge_boost != 1.0:
                weight_np[-structure_seed_count:] *= np.float32(structure_merge_boost)
            flat_weight = weight_np.reshape(-1)
            displacement_sum = np.zeros_like(input_np)
            weight_sum = np.zeros((n_points,), dtype=np.float32)
            np.add.at(displacement_sum, flat_idx, flat_disp * flat_weight[:, None])
            np.add.at(weight_sum, flat_idx, flat_weight)
            covered = weight_sum > 0
            weighted_np[covered] += (
                displacement_sum[covered] / weight_sum[covered, None]
            )

        need_patch_stats = merge_mode == "adaptive" or (
            diagnostics is not None and record_diagnostics
        )
        coverage = None
        disagreement = None
        if need_patch_stats:
            coverage = np.bincount(flat_idx, minlength=n_points).astype("int32")
            disp_sum = np.zeros((n_points, 3), dtype=np.float32)
            disp_norm2_sum = np.zeros((n_points,), dtype=np.float32)
            np.add.at(disp_sum, flat_idx, flat_disp)
            np.add.at(disp_norm2_sum, flat_idx, (flat_disp * flat_disp).sum(axis=1))
            safe_coverage = np.maximum(coverage, 1).astype(np.float32)
            mean_disp = disp_sum / safe_coverage[:, None]
            disagreement2 = (
                disp_norm2_sum / safe_coverage - (mean_disp * mean_disp).sum(axis=1)
            )
            disagreement = np.sqrt(np.maximum(disagreement2, 0.0)).astype(np.float32)

        adaptive_gate = None
        if merge_mode == "hard":
            out_np = hard_np
        elif merge_mode == "weighted":
            out_np = weighted_np
        else:
            # Trust overlap averaging where patch predictions agree, but fall
            # back to the nearest-center hard assignment at risky seams.
            confidence = np.exp(
                -np.square(disagreement / np.float32(merge_adaptive_tau))
            ).astype(np.float32)
            adaptive_gate = np.float32(merge_adaptive_alpha_max) * confidence
            out_np = hard_np + adaptive_gate[:, None] * (weighted_np - hard_np)

        if diagnostics is not None and record_diagnostics:
            diagnostics["coverage_count"] = coverage
            diagnostics["patch_disagreement_rms"] = disagreement
            diagnostics["patch_centers"] = seed_pnts[0].numpy().astype("float32")
            diagnostics["structure_seed_count"] = np.asarray(
                structure_seed_count, dtype=np.int32
            )
            # Adaptive inference already computes both merge endpoints.  Keep
            # them in diagnostics so alpha/tau/beta-independent gates can be
            # swept offline without another expensive network forward pass.
            if hard_np is not None:
                diagnostics["hard_prediction"] = hard_np.astype("float32")
            if weighted_np is not None:
                diagnostics["weighted_prediction"] = weighted_np.astype("float32")
            if adaptive_gate is not None:
                diagnostics["adaptive_gate"] = adaptive_gate
            if merge_mode in {"hard", "adaptive"}:
                selected_distance = np.full((n_points,), np.nan, dtype=np.float32)
                for patch_id in range(num_patches):
                    global_idx = idx_np[patch_id]
                    selected = best_np[global_idx] == patch_id
                    selected_distance[global_idx[selected]] = dist_np[patch_id, selected]
                diagnostics["hard_assignment"] = best_np.astype("int32")
                diagnostics["selected_center_distance"] = selected_distance
        return jt.array(out_np)

    denoised = merge_patches(patches_denoised, record_diagnostics=True)
    if not return_intermediates:
        return denoised
    patches_intermediate = jt.concat(intermediate_chunks, dim=0) + seed_expand
    intermediate = merge_patches(patches_intermediate)
    return denoised, intermediate
