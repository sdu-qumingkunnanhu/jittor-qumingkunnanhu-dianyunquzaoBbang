import jittor as jt
from jittor import Function


def _fps_indices_cuda(points: jt.Var, num_points: int) -> jt.Var:
    batch_size, num_input, _ = points.shape
    indices = jt.empty((batch_size, num_points), dtype="int32")
    temp = jt.empty((batch_size, num_input), dtype="float32")

    cpu_src = r"""
    const int bsz = in0->shape[0];
    const int n = in0->shape[1];
    const int m = out0->shape[1];
    for (int b = 0; b < bsz; ++b) {
        float* temp_b = out1_p + b * n;
        int* idx_b = out0_p + b * m;
        const float* points_b = in0_p + b * n * 3;
        for (int i = 0; i < n; ++i) {
            temp_b[i] = 1e10f;
        }
        int old = 0;
        if (m > 0) {
            idx_b[0] = 0;
        }
        for (int j = 1; j < m; ++j) {
            float best = -1.0f;
            int best_idx = 0;
            const float x1 = points_b[old * 3 + 0];
            const float y1 = points_b[old * 3 + 1];
            const float z1 = points_b[old * 3 + 2];
            for (int k = 0; k < n; ++k) {
                const float x2 = points_b[k * 3 + 0];
                const float y2 = points_b[k * 3 + 1];
                const float z2 = points_b[k * 3 + 2];
                const float dx = x2 - x1;
                const float dy = y2 - y1;
                const float dz = z2 - z1;
                const float d = dx * dx + dy * dy + dz * dz;
                const float d2 = d < temp_b[k] ? d : temp_b[k];
                temp_b[k] = d2;
                if (d2 > best) {
                    best = d2;
                    best_idx = k;
                }
            }
            old = best_idx;
            idx_b[j] = old;
        }
    }
    """

    cuda_header = r"""
    __device__ __forceinline__ void fps_update(float* dists, int* dists_i, int idx1, int idx2) {
        const float v1 = dists[idx1];
        const float v2 = dists[idx2];
        const int i1 = dists_i[idx1];
        const int i2 = dists_i[idx2];
        dists[idx1] = v1 > v2 ? v1 : v2;
        dists_i[idx1] = v2 > v1 ? i2 : i1;
    }

    template <unsigned int block_size>
    __global__ void fps_kernel(const float* __restrict__ points,
                               int* __restrict__ idx,
                               float* __restrict__ temp,
                               int n,
                               int m) {
        __shared__ float dists[block_size];
        __shared__ int dists_i[block_size];

        const int b = blockIdx.x;
        const int tid = threadIdx.x;
        const int stride = block_size;
        const float* points_b = points + b * n * 3;
        float* temp_b = temp + b * n;
        int* idx_b = idx + b * m;

        for (int k = tid; k < n; k += stride) {
            temp_b[k] = 1e10f;
        }
        if (tid == 0 && m > 0) {
            idx_b[0] = 0;
        }
        __syncthreads();

        int old = 0;
        for (int j = 1; j < m; ++j) {
            int best_i = 0;
            float best = -1.0f;
            const float x1 = points_b[old * 3 + 0];
            const float y1 = points_b[old * 3 + 1];
            const float z1 = points_b[old * 3 + 2];
            for (int k = tid; k < n; k += stride) {
                const float x2 = points_b[k * 3 + 0];
                const float y2 = points_b[k * 3 + 1];
                const float z2 = points_b[k * 3 + 2];
                const float dx = x2 - x1;
                const float dy = y2 - y1;
                const float dz = z2 - z1;
                const float d = dx * dx + dy * dy + dz * dz;
                const float d2 = d < temp_b[k] ? d : temp_b[k];
                temp_b[k] = d2;
                if (d2 > best) {
                    best = d2;
                    best_i = k;
                }
            }

            dists[tid] = best;
            dists_i[tid] = best_i;
            __syncthreads();

            if (block_size >= 1024) { if (tid < 512) fps_update(dists, dists_i, tid, tid + 512); __syncthreads(); }
            if (block_size >= 512) { if (tid < 256) fps_update(dists, dists_i, tid, tid + 256); __syncthreads(); }
            if (block_size >= 256) { if (tid < 128) fps_update(dists, dists_i, tid, tid + 128); __syncthreads(); }
            if (block_size >= 128) { if (tid < 64) fps_update(dists, dists_i, tid, tid + 64); __syncthreads(); }
            if (block_size >= 64) { if (tid < 32) fps_update(dists, dists_i, tid, tid + 32); __syncthreads(); }
            if (block_size >= 32) { if (tid < 16) fps_update(dists, dists_i, tid, tid + 16); __syncthreads(); }
            if (block_size >= 16) { if (tid < 8) fps_update(dists, dists_i, tid, tid + 8); __syncthreads(); }
            if (block_size >= 8) { if (tid < 4) fps_update(dists, dists_i, tid, tid + 4); __syncthreads(); }
            if (block_size >= 4) { if (tid < 2) fps_update(dists, dists_i, tid, tid + 2); __syncthreads(); }
            if (block_size >= 2) { if (tid < 1) fps_update(dists, dists_i, tid, tid + 1); __syncthreads(); }

            old = dists_i[0];
            if (tid == 0) {
                idx_b[j] = old;
            }
            __syncthreads();
        }
    }
    """

    cuda_src = r"""
    const int n = in0->shape[1];
    const int m = out0->shape[1];
    const int bsz = in0->shape[0];
    const int threads = 512;
    fps_kernel<512><<<bsz, threads>>>(in0_p, out0_p, out1_p, n, m);
    """

    idx, _ = jt.code([points], [indices, temp], cpu_src=cpu_src, cuda_src=cuda_src, cuda_header=cuda_header)
    return idx


def farthest_point_sampling(points: jt.Var, num_points: int):
    """
    Batched farthest point sampling matching PGD/pointops behavior:
    the first sampled point of every batch item is index 0.
    """
    batch_size, _, channels = points.shape
    if num_points <= 0:
        empty_idx = jt.empty((batch_size, 0), dtype="int32")
        return jt.empty((batch_size, 0, channels), dtype=points.dtype), empty_idx

    idx = _fps_indices_cuda(points.float32(), int(num_points))
    sampled = batch_gather(points, idx)
    return sampled, idx


def batch_gather(values: jt.Var, indices: jt.Var) -> jt.Var:
    """
    Gather values on dimension 1 with batched indices.
    values: (B, N, C)
    indices: (B, M) or (B, M, K)
    """
    if len(indices.shape) == 2:
        return values.reindex(
            (indices.shape[0], indices.shape[1], values.shape[-1]),
            ["i0", "@e0(i0,i1)", "i2"],
            extras=[indices],
        )
    if len(indices.shape) == 3:
        return values.reindex(
            (indices.shape[0], indices.shape[1], indices.shape[2], values.shape[-1]),
            ["i0", "@e0(i0,i1,i2)", "i3"],
            extras=[indices],
        )
    raise ValueError(f"indices must be 2D or 3D, got shape {indices.shape}")


class _NearestNeighborDistance(Function):
    def execute(self, p1: jt.Var, p2: jt.Var):
        p1 = p1.float32()
        p2 = p2.float32()
        bsz, n1, _ = p1.shape
        _, n2, _ = p2.shape
        d1 = jt.empty((bsz, n1), dtype="float32")
        d2 = jt.empty((bsz, n2), dtype="float32")
        idx1 = jt.empty((bsz, n1), dtype="int32")
        idx2 = jt.empty((bsz, n2), dtype="int32")

        cpu_src = r"""
        const int bsz = in0->shape[0];
        const int n1 = in0->shape[1];
        const int n2 = in1->shape[1];
        for (int b = 0; b < bsz; ++b) {
            const float* p1_b = in0_p + b * n1 * 3;
            const float* p2_b = in1_p + b * n2 * 3;
            float* d1_b = out0_p + b * n1;
            float* d2_b = out1_p + b * n2;
            int* idx1_b = out2_p + b * n1;
            int* idx2_b = out3_p + b * n2;
            for (int i = 0; i < n1; ++i) {
                float best = 1e30f;
                int best_i = 0;
                const float x1 = p1_b[i * 3 + 0];
                const float y1 = p1_b[i * 3 + 1];
                const float z1 = p1_b[i * 3 + 2];
                for (int j = 0; j < n2; ++j) {
                    const float dx = x1 - p2_b[j * 3 + 0];
                    const float dy = y1 - p2_b[j * 3 + 1];
                    const float dz = z1 - p2_b[j * 3 + 2];
                    const float dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 < best) {
                        best = dist2;
                        best_i = j;
                    }
                }
                d1_b[i] = sqrtf(best);
                idx1_b[i] = best_i;
            }
            for (int i = 0; i < n2; ++i) {
                float best = 1e30f;
                int best_i = 0;
                const float x1 = p2_b[i * 3 + 0];
                const float y1 = p2_b[i * 3 + 1];
                const float z1 = p2_b[i * 3 + 2];
                for (int j = 0; j < n1; ++j) {
                    const float dx = x1 - p1_b[j * 3 + 0];
                    const float dy = y1 - p1_b[j * 3 + 1];
                    const float dz = z1 - p1_b[j * 3 + 2];
                    const float dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 < best) {
                        best = dist2;
                        best_i = j;
                    }
                }
                d2_b[i] = sqrtf(best);
                idx2_b[i] = best_i;
            }
        }
        """

        cuda_header = r"""
        template <int tile_size>
        __global__ void nearest_forward_kernel(const float* __restrict__ query,
                                               const float* __restrict__ known,
                                               float* __restrict__ dist,
                                               int* __restrict__ idx,
                                               int bsz,
                                               int nq,
                                               int nk) {
            __shared__ float known_tile[tile_size * 3];

            const int b = blockIdx.x;
            const int i = blockIdx.y * blockDim.x + threadIdx.x;
            const int tid = threadIdx.x;

            float x1 = 0.0f;
            float y1 = 0.0f;
            float z1 = 0.0f;
            bool valid = b < bsz && i < nq;
            if (valid) {
                const float* query_b = query + b * nq * 3;
                x1 = query_b[i * 3 + 0];
                y1 = query_b[i * 3 + 1];
                z1 = query_b[i * 3 + 2];
            }

            float best = 1e30f;
            int best_i = 0;
            for (int base = 0; base < nk; base += tile_size) {
                const int tile_count = min(tile_size, nk - base);
                const float* known_b = known + b * nk * 3;
                for (int t = tid; t < tile_count * 3; t += blockDim.x) {
                    known_tile[t] = known_b[base * 3 + t];
                }
                __syncthreads();

                if (valid) {
                    int k = 0;
                    for (; k + 4 <= tile_count; k += 4) {
                        const float dx0 = x1 - known_tile[(k + 0) * 3 + 0];
                        const float dy0 = y1 - known_tile[(k + 0) * 3 + 1];
                        const float dz0 = z1 - known_tile[(k + 0) * 3 + 2];
                        const float d0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
                        if (d0 < best) { best = d0; best_i = base + k + 0; }

                        const float dx1 = x1 - known_tile[(k + 1) * 3 + 0];
                        const float dy1 = y1 - known_tile[(k + 1) * 3 + 1];
                        const float dz1 = z1 - known_tile[(k + 1) * 3 + 2];
                        const float d1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
                        if (d1 < best) { best = d1; best_i = base + k + 1; }

                        const float dx2 = x1 - known_tile[(k + 2) * 3 + 0];
                        const float dy2 = y1 - known_tile[(k + 2) * 3 + 1];
                        const float dz2 = z1 - known_tile[(k + 2) * 3 + 2];
                        const float d2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                        if (d2 < best) { best = d2; best_i = base + k + 2; }

                        const float dx3 = x1 - known_tile[(k + 3) * 3 + 0];
                        const float dy3 = y1 - known_tile[(k + 3) * 3 + 1];
                        const float dz3 = z1 - known_tile[(k + 3) * 3 + 2];
                        const float d3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;
                        if (d3 < best) { best = d3; best_i = base + k + 3; }
                    }
                    for (; k < tile_count; ++k) {
                        const float dx = x1 - known_tile[k * 3 + 0];
                        const float dy = y1 - known_tile[k * 3 + 1];
                        const float dz = z1 - known_tile[k * 3 + 2];
                        const float d = dx * dx + dy * dy + dz * dz;
                        if (d < best) {
                            best = d;
                            best_i = base + k;
                        }
                    }
                }
                __syncthreads();
            }

            if (valid) {
                const int linear = b * nq + i;
                dist[linear] = sqrtf(best);
                idx[linear] = best_i;
            }
        }
        """

        cuda_src = r"""
        const int bsz = in0->shape[0];
        const int n1 = in0->shape[1];
        const int n2 = in1->shape[1];
        const int threads = 256;
        dim3 grid1(bsz, (n1 + threads - 1) / threads);
        dim3 grid2(bsz, (n2 + threads - 1) / threads);
        nearest_forward_kernel<512><<<grid1, threads>>>(
            in0_p, in1_p, out0_p, out2_p, bsz, n1, n2);
        nearest_forward_kernel<512><<<grid2, threads>>>(
            in1_p, in0_p, out1_p, out3_p, bsz, n2, n1);
        """

        d1, d2, idx1, idx2 = jt.code(
            [p1, p2],
            [d1, d2, idx1, idx2],
            cpu_src=cpu_src,
            cuda_src=cuda_src,
            cuda_header=cuda_header,
        )
        self.p1 = p1
        self.p2 = p2
        self.idx1 = idx1
        self.idx2 = idx2
        return d1, d2

    def grad(self, grad_d1, grad_d2):
        p1 = self.p1
        p2 = self.p2
        idx1 = self.idx1
        idx2 = self.idx2
        if grad_d1 is None:
            grad_d1 = jt.zeros((p1.shape[0], p1.shape[1]), dtype=p1.dtype)
        if grad_d2 is None:
            grad_d2 = jt.zeros((p2.shape[0], p2.shape[1]), dtype=p2.dtype)

        grad_p1 = jt.empty(p1.shape, dtype=p1.dtype)
        grad_p2 = jt.empty(p2.shape, dtype=p2.dtype)

        cpu_src = r"""
        const int bsz = in0->shape[0];
        const int n1 = in0->shape[1];
        const int n2 = in1->shape[1];
        for (int i = 0; i < bsz * n1 * 3; ++i) out0_p[i] = 0.0f;
        for (int i = 0; i < bsz * n2 * 3; ++i) out1_p[i] = 0.0f;
        for (int b = 0; b < bsz; ++b) {
            const float* p1_b = in0_p + b * n1 * 3;
            const float* p2_b = in1_p + b * n2 * 3;
            const int* idx1_b = in2_p + b * n1;
            const int* idx2_b = in3_p + b * n2;
            const float* gd1_b = in4_p + b * n1;
            const float* gd2_b = in5_p + b * n2;
            float* gp1_b = out0_p + b * n1 * 3;
            float* gp2_b = out1_p + b * n2 * 3;
            for (int i = 0; i < n1; ++i) {
                const int j = idx1_b[i];
                const float dx = p1_b[i * 3 + 0] - p2_b[j * 3 + 0];
                const float dy = p1_b[i * 3 + 1] - p2_b[j * 3 + 1];
                const float dz = p1_b[i * 3 + 2] - p2_b[j * 3 + 2];
                const float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-12f);
                const float g = gd1_b[i] / dist;
                gp1_b[i * 3 + 0] += g * dx;
                gp1_b[i * 3 + 1] += g * dy;
                gp1_b[i * 3 + 2] += g * dz;
                gp2_b[j * 3 + 0] -= g * dx;
                gp2_b[j * 3 + 1] -= g * dy;
                gp2_b[j * 3 + 2] -= g * dz;
            }
            for (int i = 0; i < n2; ++i) {
                const int j = idx2_b[i];
                const float dx = p2_b[i * 3 + 0] - p1_b[j * 3 + 0];
                const float dy = p2_b[i * 3 + 1] - p1_b[j * 3 + 1];
                const float dz = p2_b[i * 3 + 2] - p1_b[j * 3 + 2];
                const float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-12f);
                const float g = gd2_b[i] / dist;
                gp2_b[i * 3 + 0] += g * dx;
                gp2_b[i * 3 + 1] += g * dy;
                gp2_b[i * 3 + 2] += g * dz;
                gp1_b[j * 3 + 0] -= g * dx;
                gp1_b[j * 3 + 1] -= g * dy;
                gp1_b[j * 3 + 2] -= g * dz;
            }
        }
        """

        cuda_header = r"""
        __global__ void nearest_grad_zero_kernel(float* p, int total) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < total) p[i] = 0.0f;
        }

        __global__ void nearest_grad_kernel(const float* __restrict__ query,
                                            const float* __restrict__ known,
                                            const int* __restrict__ idx,
                                            const float* __restrict__ grad_dist,
                                            float* __restrict__ grad_query,
                                            float* __restrict__ grad_known,
                                            int bsz,
                                            int nq,
                                            int nk) {
            const int linear = blockIdx.x * blockDim.x + threadIdx.x;
            const int total = bsz * nq;
            if (linear >= total) return;
            const int b = linear / nq;
            const int i = linear - b * nq;
            const int j = idx[linear];
            const float* query_b = query + b * nq * 3;
            const float* known_b = known + b * nk * 3;
            float* grad_query_b = grad_query + b * nq * 3;
            float* grad_known_b = grad_known + b * nk * 3;

            const float dx = query_b[i * 3 + 0] - known_b[j * 3 + 0];
            const float dy = query_b[i * 3 + 1] - known_b[j * 3 + 1];
            const float dz = query_b[i * 3 + 2] - known_b[j * 3 + 2];
            const float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-12f);
            const float g = grad_dist[linear] / dist;
            const float gx = g * dx;
            const float gy = g * dy;
            const float gz = g * dz;
            atomicAdd(grad_query_b + i * 3 + 0, gx);
            atomicAdd(grad_query_b + i * 3 + 1, gy);
            atomicAdd(grad_query_b + i * 3 + 2, gz);
            atomicAdd(grad_known_b + j * 3 + 0, -gx);
            atomicAdd(grad_known_b + j * 3 + 1, -gy);
            atomicAdd(grad_known_b + j * 3 + 2, -gz);
        }
        """

        cuda_src = r"""
        const int bsz = in0->shape[0];
        const int n1 = in0->shape[1];
        const int n2 = in1->shape[1];
        const int threads = 256;
        nearest_grad_zero_kernel<<<(bsz * n1 * 3 + threads - 1) / threads, threads>>>(out0_p, bsz * n1 * 3);
        nearest_grad_zero_kernel<<<(bsz * n2 * 3 + threads - 1) / threads, threads>>>(out1_p, bsz * n2 * 3);
        nearest_grad_kernel<<<(bsz * n1 + threads - 1) / threads, threads>>>(
            in0_p, in1_p, in2_p, in4_p, out0_p, out1_p, bsz, n1, n2);
        nearest_grad_kernel<<<(bsz * n2 + threads - 1) / threads, threads>>>(
            in1_p, in0_p, in3_p, in5_p, out1_p, out0_p, bsz, n2, n1);
        """

        grad_p1, grad_p2 = jt.code(
            [p1, p2, idx1, idx2, grad_d1, grad_d2],
            [grad_p1, grad_p2],
            cpu_src=cpu_src,
            cuda_src=cuda_src,
            cuda_header=cuda_header,
        )
        return grad_p1, grad_p2


def nearest_neighbor_distance(p1: jt.Var, p2: jt.Var):
    return _NearestNeighborDistance.apply(p1, p2)
