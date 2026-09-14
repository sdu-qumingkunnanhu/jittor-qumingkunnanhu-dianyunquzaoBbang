import math
from typing import Tuple

import jittor as jt

from src.utils.pointops import batch_gather, farthest_point_sampling


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


def patch_based_denoise(model, pcl_noisy: jt.Var, patch_size=1000, seed_k=6, seed_k_alpha=10):
    assert len(pcl_noisy.shape) == 2, "input point cloud must be (N, 3)"
    n_points, _ = pcl_noisy.shape
    num_patches = max(1, int(seed_k * n_points / patch_size))
    pcl_batch = pcl_noisy.unsqueeze(0)
    seed_pnts, _ = farthest_point_sampling(pcl_batch, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_batch, patch_size)

    patches = patches[0]
    patch_dists = patch_dists[0]
    point_idxs = point_idxs[0]
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches_centered = patches - seed_expand

    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    all_dists = jt.ones((num_patches, n_points)) * 1e10
    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]
    weights = jt.exp(-all_dists)
    best_patch, _ = jt.argmax(weights, dim=0)

    denoised_chunks = []
    step = max(1, int(math.ceil(n_points / (seed_k_alpha * patch_size))))
    i = 0
    while i < num_patches:
        denoised_chunks.append(model.denoise_langevin_dynamics(patches_centered[i:i + step]))
        i += step
    patches_denoised = jt.concat(denoised_chunks, dim=0) + seed_expand

    out_np = pcl_noisy.numpy().astype("float32")
    best_np = best_patch.numpy().astype("int64")
    idx_np = point_idxs.numpy().astype("int64")
    patches_np = patches_denoised.numpy().astype("float32")
    for patch_id in range(num_patches):
        assigned = set((best_np == patch_id).nonzero()[0].tolist())
        if not assigned:
            continue
        for local_idx, global_idx in enumerate(idx_np[patch_id]):
            if int(global_idx) in assigned:
                out_np[int(global_idx)] = patches_np[patch_id, local_idx]
    return jt.array(out_np)
