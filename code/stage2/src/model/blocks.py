import os
import sys

import jittor as jt
from jittor import nn

from .ops import batch_gather, farthest_point_sampling, knn_points


_PCLIB_KNN = None
_PCLIB_IMPORT_ERROR = None
_PCLIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "PointCloudLib"))
if os.path.isdir(_PCLIB_ROOT):
    if _PCLIB_ROOT not in sys.path:
        sys.path.insert(0, _PCLIB_ROOT)
    try:
        from misc.ops import KNN as _PCLIB_KNN
    except Exception as exc:
        _PCLIB_IMPORT_ERROR = exc


def flat_bn(bn, x):
    shape = x.shape
    return bn(x.reshape(-1, shape[-1])).reshape(shape)


class MLPBN(nn.Module):
    def __init__(self, in_dim, out_dim, activation="relu", bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

    def execute(self, x):
        shape = x.shape
        x = self.linear(x.reshape(-1, shape[-1]))
        x = self.bn(x)
        if self.activation == "relu":
            x = nn.relu(x)
        elif self.activation == "leaky_relu":
            x = nn.leaky_relu(x, scale=0.2)
        return x.reshape(*shape[:-1], -1)


class RFE(nn.Module):
    def __init__(self, d_in, d_out, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.bi_1 = nn.Linear(10, 2 * d_out)
        self.bi_bn = nn.BatchNorm1d(2 * d_out)
        self.bi_2 = nn.Linear(2 * d_out, d_out)
        self.score = nn.Linear(d_out * 2, d_out * 2, bias=False)
        self.out = MLPBN(d_out * 2, d_out, activation="relu")

    def execute(self, grouped_p, grouped_x):
        center = grouped_p[:, :, 0:1, :].broadcast(grouped_p.shape)
        dist = jt.sqrt(((center - grouped_p) ** 2).sum(dim=-1, keepdims=True) + 1e-12)
        pos = jt.concat([center, grouped_p, center - grouped_p, dist], dim=-1)
        shape = pos.shape
        pos_feat = self.bi_1(pos.reshape(-1, shape[-1]))
        pos_feat = self.bi_bn(pos_feat)
        pos_feat = nn.relu(pos_feat)
        pos_feat = self.bi_2(pos_feat).reshape(*shape[:-1], -1)

        feat = jt.concat([pos_feat, grouped_x], dim=-1)
        scores = nn.softmax(self.score(feat), dim=-2)
        out = (scores * feat).sum(dim=-2)
        return self.out(out)


class MRE(nn.Module):
    def __init__(self, d_in, d_out, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.mlp0 = MLPBN(d_in, d_out // 2, activation="relu")
        self.mlp1 = MLPBN(d_out, d_out, activation="relu")
        self.mlp01 = MLPBN(d_in, d_out, activation="relu")
        self.rfe1 = RFE(d_out // 2, d_out // 2, nsample)
        self.rfe2 = RFE(d_out // 2, d_out // 2, nsample)

    def execute(self, p, x):
        x_start = x
        x = self.mlp0(x)
        _, idx, grouped_p = knn_points(p, p, self.nsample)
        grouped_x = batch_gather(x, idx)
        grouped_p = grouped_p - p.unsqueeze(2)
        x = self.rfe1(grouped_p, grouped_x)
        x_middle = x

        _, idx, grouped_p = knn_points(p, p, self.nsample)
        grouped_x = batch_gather(x, idx)
        grouped_p = grouped_p - p.unsqueeze(2)
        x = self.rfe2(grouped_p, grouped_x)
        x = jt.concat([x_middle, x], dim=-1)
        return self.mlp01(x_start) + self.mlp1(x)


class NAA(nn.Module):
    def __init__(self, dim, k=32, group_backend="jt"):
        super().__init__()
        self.dim = dim
        self.k = k
        self.group_backend = str(group_backend)
        if self.group_backend not in ("jt", "pointcloudlib"):
            raise ValueError(f"unsupported naa_group_backend: {self.group_backend}")
        if self.group_backend == "pointcloudlib":
            if _PCLIB_KNN is None:
                raise ImportError(f"PointCloudLib KNN is unavailable: {_PCLIB_IMPORT_ERROR}")
            self.pclib_knn = _PCLIB_KNN(k=self.k)
        else:
            self.pclib_knn = None
        self.pos_mlp = MLPBN(3, dim, activation="relu")
        self.attn_1 = nn.Linear(dim * 2, dim * 2)
        self.attn_2 = nn.Linear(dim * 2, dim * 2)
        self.out_mlp = MLPBN(dim * 2, dim, activation="relu")

    def _query_and_group_pointcloudlib(self, p, x, k):
        # PointCloudLib KNN follows [B, C, N] layout and returns [B, k, N].
        knn = self.pclib_knn if k == self.k else _PCLIB_KNN(k=k)
        idx = knn(p.transpose(0, 2, 1), p.transpose(0, 2, 1)).transpose(0, 2, 1)
        grouped_p = batch_gather(p, idx)
        grouped_x = batch_gather(x, idx)
        return grouped_p - p.unsqueeze(2), grouped_x

    def execute(self, p, x):
        k = min(self.k, p.shape[1])
        if self.group_backend == "pointcloudlib":
            rel_pos, neigh_x = self._query_and_group_pointcloudlib(p, x, k)
        else:
            _, idx, grouped_p = knn_points(p, p, k)
            rel_pos = grouped_p - p.unsqueeze(2)
            neigh_x = batch_gather(x, idx)
        pos_feat = self.pos_mlp(rel_pos)
        feat = jt.concat([neigh_x, pos_feat], dim=-1)
        shape = feat.shape
        score = self.attn_1(feat.reshape(-1, shape[-1]))
        score = nn.relu(score)
        score = self.attn_2(score).reshape(shape)
        weight = nn.softmax(score, dim=2)
        out = (weight * feat).sum(dim=2)
        return self.out_mlp(out)


class SELayer(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(int(dim) // max(int(reduction), 1), 1)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.scale = jt.zeros((1,))

    def execute(self, x):
        gate = x.mean(dim=1)
        gate = nn.relu(self.fc1(gate))
        gate = jt.sigmoid(self.fc2(gate)).unsqueeze(1)
        return x * (1.0 + self.scale * (gate - 0.5) * 2.0)


class DSNetNAAEncoderBlock(nn.Module):
    def __init__(self, d_in, d_out, k=32, use_se=False, se_reduction=4, group_backend="jt"):
        super().__init__()
        hidden_dim = d_out // 2
        self.mlp0 = MLPBN(d_in, hidden_dim, activation="relu")
        self.naa_1 = NAA(hidden_dim, k=k, group_backend=group_backend)
        self.naa_2 = NAA(hidden_dim, k=k, group_backend=group_backend)
        self.fuse_mlp = MLPBN(d_out, d_out, activation="relu")
        self.res_mlp = MLPBN(d_in, d_out, activation="relu")
        self.use_se = bool(use_se)
        self.se = SELayer(d_out, reduction=se_reduction) if self.use_se else None

    def execute(self, p, x):
        x_start = x
        x0 = self.mlp0(x)
        x_naa1 = self.naa_1(p, x0)
        x_naa2 = self.naa_2(p, x_naa1)
        x_fuse = jt.concat([x_naa1, x_naa2], dim=-1)
        x_fuse = self.fuse_mlp(x_fuse)
        if self.se is not None:
            x_fuse = self.se(x_fuse)
        return x_fuse + self.res_mlp(x_start)


class StartBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.mlp = MLPBN(d_in + 3, d_out, activation="leaky_relu")

    def execute(self, p, x=None):
        return p, self.mlp(p), None


class Downsampling(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        nsample,
        stride,
        encoder_type="mre",
        naa_k=32,
        naa_use_se=False,
        naa_se_reduction=4,
        downsample_method="fps",
        naa_group_backend="jt",
    ):
        super().__init__()
        self.stride = stride
        self.encoder_type = encoder_type
        self.downsample_method = str(downsample_method)
        if self.downsample_method not in ("fps", "linspace_index"):
            raise ValueError(f"unsupported downsample_method: {self.downsample_method}")
        if encoder_type == "mre":
            self.mre = MRE(d_in, d_out, nsample)
        elif encoder_type == "naa":
            self.encoder = DSNetNAAEncoderBlock(
                d_in,
                d_out,
                k=naa_k,
                use_se=naa_use_se,
                se_reduction=naa_se_reduction,
                group_backend=naa_group_backend,
            )
        else:
            raise ValueError(f"unsupported encoder_type: {encoder_type}")

    def _linspace_index_sampling(self, p, count):
        num_points = p.shape[1]
        idx = jt.linspace(0, num_points - 1, count).round().int32()
        idx = idx.unsqueeze(0).broadcast((p.shape[0], count))
        return batch_gather(p, idx.unsqueeze(-1)).squeeze(2), idx

    def execute(self, p, x):
        x = self.mre(p, x) if self.encoder_type == "mre" else self.encoder(p, x)
        count = max(1, p.shape[1] * self.stride // (self.stride + 1))
        if self.downsample_method == "fps":
            n_p, idx = farthest_point_sampling(p, count)
        else:
            n_p, idx = self._linspace_index_sampling(p, count)
        n_x = batch_gather(x, idx.unsqueeze(-1)).squeeze(2)
        return n_p, n_x, idx


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.fc2 = nn.Linear(input_dim * 2, output_dim * 2)

    def execute(self, x):
        params = self.fc2(nn.leaky_relu(self.fc1(x), scale=0.2))
        return jt.chunk(params, 2, dim=-1)


class CrossAttentionPointTransformerLayer(nn.Module):
    def __init__(self, dim, dim_dense, k_sample, attn_mlp_hidden_mult=1, num_neighbors=16):
        super().__init__()
        self.num_neighbors = num_neighbors
        out_dim = dim * (k_sample + 1) // k_sample
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, out_dim, bias=False)
        self.to_v = nn.Linear(dim, out_dim, bias=False)
        self.film_layer = FiLMLayer(dim_dense, dim)
        self.attn_1 = nn.Linear(dim, dim * attn_mlp_hidden_mult)
        self.attn_bn = nn.BatchNorm1d(dim * attn_mlp_hidden_mult)
        self.attn_2 = nn.Linear(dim * attn_mlp_hidden_mult, dim)

    def execute(self, x_e, x_r, x_d, pos, quantized_features=None):
        q = self.to_q(x_e)
        k_raw = self.to_k(x_r)
        v_raw = self.to_v(x_d)
        k_dim = (x_r.shape[1] * k_raw.shape[2]) // q.shape[1]
        v_dim = (x_d.shape[1] * v_raw.shape[2]) // q.shape[1]
        k = k_raw.reshape(q.shape[0], q.shape[1], k_dim)
        v = v_raw.reshape(q.shape[0], q.shape[1], v_dim)
        if quantized_features is not None:
            gamma, beta = self.film_layer(quantized_features)
            k = gamma * k + beta

        if self.num_neighbors is not None and self.num_neighbors < x_e.shape[1]:
            _, idx, _ = knn_points(pos, pos, self.num_neighbors)
            k = batch_gather(k, idx)
            v = batch_gather(v, idx)
            x_e_neigh = batch_gather(x_e, idx)
            qk_rel = q.unsqueeze(2) - k
        else:
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            x_e_neigh = x_e.unsqueeze(2)
            qk_rel = q.unsqueeze(2) - k

        v = v + x_e_neigh
        sim_in = qk_rel + x_e_neigh
        shape = sim_in.shape
        sim = self.attn_1(sim_in.reshape(-1, shape[-1]))
        sim = self.attn_bn(sim)
        sim = nn.relu(sim)
        sim = self.attn_2(sim).reshape(shape)
        attn = nn.softmax(sim, dim=-2)
        return (attn * v).sum(dim=-2)


def _safe_norm(x, dim=1, keepdims=True, eps=1e-6):
    return jt.norm(x, dim=dim, keepdims=keepdims).clamp(eps, 1e30)


def rotation_trick_backward(grad_output, features, quantized, eps=1e-6):
    original_shape = grad_output.shape
    g = grad_output.reshape(-1, grad_output.shape[-1])
    src = quantized.reshape(-1, quantized.shape[-1])
    tgt = features.reshape(-1, features.shape[-1])

    src_norm = _safe_norm(src, dim=1, keepdims=True, eps=eps)
    tgt_norm = _safe_norm(tgt, dim=1, keepdims=True, eps=eps)
    src_hat = src / src_norm
    tgt_hat = tgt / tgt_norm

    cos = (src_hat * tgt_hat).sum(dim=1, keepdims=True).clamp(-1.0 + eps, 1.0 - eps)
    parallel = (g * src_hat).sum(dim=1, keepdims=True) * src_hat
    perp = g - parallel

    axis = tgt_hat - cos * src_hat
    axis_norm_raw = jt.norm(axis, dim=1, keepdims=True)
    axis_norm = axis_norm_raw.clamp(eps, 1e30)
    axis_hat = axis / axis_norm

    rotated_parallel = (g * src_hat).sum(dim=1, keepdims=True) * tgt_hat
    axis_grad = (g * axis_hat).sum(dim=1, keepdims=True)
    rotated_perp_in_plane = axis_grad * (-axis_norm * src_hat + cos * axis_hat)
    perp_orth = perp - axis_grad * axis_hat
    rotated = rotated_parallel + rotated_perp_in_plane + perp_orth

    rotated = jt.where(axis_norm_raw <= eps, g, rotated)
    scale = (src_norm / tgt_norm).clamp(0.1, 10.0)
    return (rotated * scale).reshape(original_shape)


class _RotationTrickSTE(jt.Function):
    def execute(self, features, quantized):
        self.features = features
        self.quantized = quantized
        return quantized

    def grad(self, grad_output):
        return rotation_trick_backward(grad_output, self.features, self.quantized), None


def rotation_trick_ste(features, quantized):
    return _RotationTrickSTE.apply(features, quantized)


class CodebookModule(nn.Module):
    def __init__(
        self,
        feature_dim=48,
        codebook_size=128,
        momentum=0.99,
        commitment_cost=0,
        use_ema=True,
        temperature=0.1,
        reset_interval=1000,
        dead_threshold=5000,
        ste_type="identity",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size
        self.momentum = momentum
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.temperature = temperature
        self.reset_interval = reset_interval
        self.dead_threshold = dead_threshold
        self.ste_type = ste_type
        if self.ste_type not in ("identity", "rotation"):
            raise ValueError(f"unsupported VQ STE type: {self.ste_type}")
        codebook = jt.randn((codebook_size, feature_dim))
        codebook = codebook / (jt.norm(codebook, dim=1, keepdims=True) + 1e-8)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", jt.zeros((codebook_size,)))
        self.register_buffer("cluster_sum", jt.zeros((codebook_size, feature_dim)))
        self.register_buffer("usage_count", jt.zeros((codebook_size,)))
        self.register_buffer("last_usage", jt.zeros((codebook_size,)))
        self.register_buffer("step_counter", jt.zeros((1,)).int32())
        self._step_counter_py = 0

    def soft_quantize(self, features):
        feat_norm = features / (jt.norm(features, dim=1, keepdims=True) + 1e-8)
        code_norm = self.codebook / (jt.norm(self.codebook, dim=1, keepdims=True) + 1e-8)
        similarity = jt.matmul(feat_norm, code_norm.transpose())
        weights = nn.softmax(similarity / self.temperature, dim=1)
        quantized = jt.matmul(weights, self.codebook)
        return quantized, weights

    def get_codebook_features(self, indices=None, features=None):
        if indices is not None:
            return self.codebook[indices]
        if features is not None:
            feat_norm = features / (jt.norm(features, dim=1, keepdims=True) + 1e-8)
            code_norm = self.codebook / (jt.norm(self.codebook, dim=1, keepdims=True) + 1e-8)
            similarity = jt.matmul(feat_norm, code_norm.transpose())
            indices = jt.argmax(similarity, dim=1)[0]
            return self.codebook[indices], indices
        return self.codebook

    def update_codebook(self, features, weights):
        if (not self.is_training()) or (not self.use_ema):
            return
        with jt.no_grad():
            self._step_counter_py += 1
            self.step_counter.assign(self.step_counter + 1)
            new_cluster_size = weights.sum(dim=0)
            new_cluster_sum = jt.matmul(weights.transpose(), features)
            self.usage_count.assign(self.usage_count + new_cluster_size)
            used = (new_cluster_size > 0).float()
            self.last_usage.assign(used * self.step_counter.float() + (1 - used) * self.last_usage)
            self.cluster_size.assign(self.cluster_size * self.momentum + new_cluster_size * (1 - self.momentum))
            self.cluster_sum.assign(self.cluster_sum * self.momentum + new_cluster_sum * (1 - self.momentum))
            updated = self.cluster_sum / (self.cluster_size.unsqueeze(1) + 1e-5)
            mask = (self.cluster_size > 1e-5).float().unsqueeze(1)
            self.codebook.assign(updated * mask + self.codebook * (1 - mask))
            if self.reset_interval > 0 and self._step_counter_py % self.reset_interval == 0:
                self._reset_dead_codebook_vectors()

    def _reset_dead_codebook_vectors(self):
        current_step = self._step_counter_py
        unused_steps = current_step - self.last_usage
        dead_indices = jt.where(unused_steps > self.dead_threshold)[0]
        num_dead = int(dead_indices.shape[0])
        if num_dead == 0:
            return
        _, most_used_indices = jt.topk(self.usage_count, k=num_dead, dim=0, largest=True)
        for i in range(num_dead):
            dead_idx = int(dead_indices[i].item())
            used_idx = int(most_used_indices[i].item())
            new_vector = self.codebook[used_idx] + jt.randn(self.codebook[used_idx].shape) * 0.1
            new_vector = new_vector / (jt.norm(new_vector, dim=0, keepdims=True) + 1e-8)
            self.codebook[dead_idx] = new_vector
            self.cluster_size[dead_idx] = 0
            self.cluster_sum[dead_idx] = 0
            self.usage_count[dead_idx] = 0
            self.last_usage[dead_idx] = current_step

    def execute(self, features, calculate_commitment_loss=False, ste_type=None):
        ste_type = self.ste_type if ste_type is None else ste_type
        if ste_type not in ("identity", "rotation"):
            raise ValueError(f"unsupported VQ STE type: {ste_type}")

        quantized, weights = self.soft_quantize(features)
        self.update_codebook(features, weights)
        commitment = jt.array(0.0)
        if self.is_training() and calculate_commitment_loss:
            commitment = ((quantized.stop_grad() - features) ** 2).mean() * self.commitment_cost

        if ste_type == "identity":
            output = features + (quantized - features).stop_grad()
        else:
            output = rotation_trick_ste(features, quantized)
        return output, commitment


class Upsampling(nn.Module):
    def __init__(
        self,
        d_in_sparse_fusion,
        d_out,
        nsample,
        stride,
        attn_mlp_hidden_mult=1,
        num_neighbors=16,
        interpolation_k=8,
    ):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_fusion
        self.stride = stride
        self.interpolation_k = interpolation_k
        self.linear_upsample = nn.Linear(d_in_sparse + d_in_dense, d_in_sparse)
        self.cross = CrossAttentionPointTransformerLayer(
            dim=d_in_sparse,
            dim_dense=d_in_dense,
            k_sample=stride,
            attn_mlp_hidden_mult=attn_mlp_hidden_mult,
            num_neighbors=num_neighbors,
        )
        self.mlp = MLPBN(d_in_sparse + d_in_dense, d_out, activation="relu")

    def interpolate(self, p_sparse, p_dense, x_sparse, k=8):
        dists, idx, _ = knn_points(p_dense, p_sparse, min(k, p_sparse.shape[1]))
        grouped_x = batch_gather(x_sparse, idx)
        weights = 1.0 / (dists + 1e-10)
        weights = weights / weights.sum(dim=-1, keepdims=True)
        return (grouped_x * weights.unsqueeze(-1)).sum(dim=2)

    def execute(
        self,
        p1,
        x1,
        idx,
        p2,
        x2,
        codebook=None,
        calculate_commitment_loss_for_block=False,
        vq_ste="identity",
    ):
        x2_interp = self.interpolate(p2, p1, x2, k=self.interpolation_k)
        u_query = self.linear_upsample(jt.concat([x1, x2_interp], dim=-1))
        commitment = jt.array(0.0)
        if codebook is not None:
            flat = x1.reshape(-1, x1.shape[-1])
            q_flat, commitment = codebook(
                flat,
                calculate_commitment_loss_for_block,
                ste_type=vq_ste,
            )
            x1 = q_flat.reshape(x1.shape)
        x1_enhance = self.cross(u_query, x2, x2, p1, x1)
        x = self.mlp(jt.concat([x1_enhance, x1], dim=-1))
        return p1, x, commitment
