import jittor as jt
from jittor import nn

from .blocks import CodebookModule, Downsampling, StartBlock, Upsampling


def _parse_naa_se_layers(layers):
    if layers is None or layers == "":
        return {1, 2, 3, 4}
    if isinstance(layers, int):
        if layers < 0 or layers > 4:
            raise ValueError(f"naa_se_layers int must be in [0, 4], got {layers}")
        return set(range(1, layers + 1))
    if isinstance(layers, (list, tuple, set)):
        parsed = {int(x) for x in layers}
    else:
        text = str(layers).strip()
        if text.isdigit():
            value = int(text)
            if value < 0 or value > 4:
                raise ValueError(f"naa_se_layers count must be in [0, 4], got {text}")
            return set(range(1, value + 1))
        parsed = {int(part.strip()) for part in text.split(",") if part.strip()}
    invalid = sorted(x for x in parsed if x < 1 or x > 4)
    if invalid:
        raise ValueError(f"naa_se_layers must be in [1, 4], got {invalid}")
    return parsed


class FeatureExtraction(nn.Module):
    def __init__(
        self,
        d_in=0,
        d_out=32,
        n_cls=3,
        nsample=16,
        stride_list=None,
        architecture=None,
        stride_dim_list=None,
        codebook_specs=None,
        codebook_commitment_cost=0,
        codebook_temperature=0.1,
        codebook_momentum=0.99,
        codebook_use_ema=True,
        codebook_reset_interval=1000,
        codebook_dead_threshold=5000,
        attn_mlp_hidden_mult=1,
        num_neighbors=16,
        interpolation_k=8,
        encoder_type="mre",
        naa_k=32,
        vq_ste="identity",
        naa_use_se=False,
        naa_se_reduction=4,
        naa_se_layers=None,
        downsample_method="fps",
        naa_group_backend="jt",
    ):
        super().__init__()
        if stride_list is None:
            stride_list = [4, 3, 2, 1]
        if architecture is None:
            architecture = [
                "startblock",
                "downsample",
                "downsample",
                "downsample",
                "downsample",
                "upsample",
                "upsample",
                "upsample",
                "upsample",
            ]
        if stride_dim_list is None:
            stride_dim_list = [1.5, 1.5, 1.5, 1.5]
        if codebook_specs is None:
            codebook_specs = [
                {"feature_dim": 108, "codebook_size": 512},
                {"feature_dim": 72, "codebook_size": 384},
                {"feature_dim": 48, "codebook_size": 256},
                {"feature_dim": 32, "codebook_size": 192},
            ]

        self.vq_ste = vq_ste
        if self.vq_ste not in ("identity", "rotation"):
            raise ValueError(f"unsupported vq_ste: {self.vq_ste}")

        self.naa_use_se = bool(naa_use_se) if encoder_type == "naa" else False
        self.naa_se_reduction = naa_se_reduction
        self.naa_se_layers = _parse_naa_se_layers(naa_se_layers) if self.naa_use_se else set()
        self.downsample_method = str(downsample_method)
        if self.downsample_method not in ("fps", "linspace_index"):
            raise ValueError(f"unsupported downsample_method: {self.downsample_method}")
        self.naa_group_backend = str(naa_group_backend)
        if self.naa_group_backend not in ("jt", "pointcloudlib"):
            raise ValueError(f"unsupported naa_group_backend: {self.naa_group_backend}")

        d_prev = d_in
        stride_idx = 0
        downsample_layer_idx = 0
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []

        for block_name in architecture:
            if block_name == "downsample":
                downsample_layer_idx += 1
                self.encoder_skip_dims.append(d_prev)
                stride = stride_list[stride_idx]
                d_out = int(d_out * stride_dim_list[stride_idx])
                stride_idx += 1
                self.encoder_blocks.append(
                    Downsampling(
                        d_prev,
                        d_out,
                        nsample,
                        stride,
                        encoder_type=encoder_type,
                        naa_k=naa_k,
                        naa_use_se=self.naa_use_se and downsample_layer_idx in self.naa_se_layers,
                        naa_se_reduction=self.naa_se_reduction,
                        downsample_method=self.downsample_method,
                        naa_group_backend=self.naa_group_backend,
                    )
                )
            elif block_name == "upsample":
                stride_idx -= 1
                stride = stride_list[stride_idx]
                skip_dim = self.encoder_skip_dims.pop()
                d_out = skip_dim
                self.decoder_blocks.append(
                    Upsampling(
                        [d_prev, skip_dim],
                        d_out,
                        nsample,
                        stride,
                        attn_mlp_hidden_mult=attn_mlp_hidden_mult,
                        num_neighbors=num_neighbors,
                        interpolation_k=interpolation_k,
                    )
                )
            else:
                self.encoder_blocks.append(StartBlock(d_prev, d_out, nsample, 1))
            d_prev = d_out

        self.linear0_1 = nn.Linear(d_out, 128, bias=False)
        self.linear0_2 = nn.Linear(128, 64)
        self.linear0_3 = nn.Linear(64, n_cls)

        self.codebooks = nn.ModuleList()
        for spec in codebook_specs:
            self.codebooks.append(
                CodebookModule(
                    feature_dim=int(spec["feature_dim"]),
                    codebook_size=int(spec["codebook_size"]),
                    momentum=float(spec.get("momentum", codebook_momentum)),
                    commitment_cost=float(spec.get("commitment_cost", codebook_commitment_cost)),
                    use_ema=bool(spec.get("use_ema", codebook_use_ema)),
                    temperature=float(spec.get("temperature", codebook_temperature)),
                    reset_interval=int(spec.get("reset_interval", codebook_reset_interval)),
                    dead_threshold=int(spec.get("dead_threshold", codebook_dead_threshold)),
                    ste_type=str(spec.get("ste_type", self.vq_ste)),
                )
            )

    def execute(self, p, calculate_commitment_losses=False):
        p_from_encoder = []
        x_from_encoder = []
        idx_from_encoder = []
        commitment = jt.array(0.0)
        x = None

        for block in self.encoder_blocks:
            p, x, idx = block(p, x)
            p_from_encoder.append(p)
            x_from_encoder.append(x)
            idx_from_encoder.append(idx)

        x = x_from_encoder.pop()
        p = p_from_encoder.pop()
        idx_from_encoder.pop()

        for block_i, block in enumerate(self.decoder_blocks):
            x_skip = x_from_encoder.pop()
            p_skip = p_from_encoder.pop()
            idx_skip = idx_from_encoder.pop()
            codebook = self.codebooks[block_i]
            p, x, block_commitment = block(
                p1=p_skip,
                x1=x_skip,
                idx=idx_skip,
                p2=p,
                x2=x,
                codebook=codebook,
                calculate_commitment_loss_for_block=calculate_commitment_losses,
                vq_ste=self.vq_ste,
            )
            commitment = commitment + block_commitment

        x = nn.relu(self.linear0_1(x))
        x = nn.relu(self.linear0_2(x))
        x = jt.tanh(self.linear0_3(x))
        return x, commitment
