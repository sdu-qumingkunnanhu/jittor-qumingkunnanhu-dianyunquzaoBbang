from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import FeatureExtraction
from .infocd import calc_cd_like_InfoV2
from .ops import chamfer_distance_unit_sphere, patch_based_denoise
from .spec import ModelSpec
from ..data.asset import Asset


class PGDModel(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.patch_size = int(cfg.get("patch_size", 1000))
        self.seed_k = int(cfg.get("seed_k", 6))
        self.seed_k_alpha = int(cfg.get("seed_k_alpha", 10))
        self.niters = int(cfg.get("niters", 1))
        self.feature_nets = FeatureExtraction(
            d_in=int(cfg.get("d_in", 0)),
            d_out=int(cfg.get("d_out", 32)),
            n_cls=int(cfg.get("n_cls", 3)),
            nsample=int(cfg.get("nsample", 16)),
            stride_list=list(cfg.get("stride_list", [4, 3, 2, 1])),
            architecture=list(cfg.get("architecture", [])) or None,
            stride_dim_list=list(cfg.get("stride_dim_list", [])) or None,
            codebook_specs=list(cfg.get("codebook_specs", [])) or None,
            codebook_commitment_cost=float(cfg.get("codebook_commitment_cost", 0.0)),
            codebook_temperature=float(cfg.get("codebook_temperature", 0.1)),
            codebook_momentum=float(cfg.get("codebook_momentum", 0.99)),
            codebook_use_ema=bool(cfg.get("codebook_use_ema", True)),
            codebook_reset_interval=int(cfg.get("codebook_reset_interval", 1000)),
            codebook_dead_threshold=int(cfg.get("codebook_dead_threshold", 5000)),
            attn_mlp_hidden_mult=int(cfg.get("attn_mlp_hidden_mult", 1)),
            num_neighbors=int(cfg.get("num_neighbors", 16)),
            interpolation_k=int(cfg.get("interpolation_k", 8)),
            encoder_type=str(cfg.get("encoder_type", "mre")),
            naa_k=int(cfg.get("naa_k", 32)),
            vq_ste=str(cfg.get("vq_ste", "identity")),
            naa_use_se=bool(cfg.get("naa_use_se", False)),
            naa_se_reduction=int(cfg.get("naa_se_reduction", 4)),
            naa_se_layers=cfg.get("naa_se_layers", None),
            downsample_method=str(cfg.get("downsample_method", "fps")),
            naa_group_backend=str(cfg.get("naa_group_backend", "jt")),
        )

    def named_parameters(self, recurse=True):
        return [
            (name, param)
            for name, param in super().named_parameters(recurse=recurse)
            if ".codebooks." not in name and not name.startswith("feature_nets.codebooks.")
        ]

    def parameters(self, recurse=True):
        return [param for _, param in self.named_parameters(recurse=recurse)]

    def get_supervised_loss_with_commitment(self, pc_noisy, pc_clean):
        pred_disp, commitment_loss = self.feature_nets(
            pc_noisy,
            calculate_commitment_losses=True,
        )
        pred_pc = pc_noisy + pred_disp
        main_loss = calc_cd_like_InfoV2(pred_pc, pc_clean)
        return main_loss, commitment_loss, pred_pc, pc_clean

    def training_step(self, batch: Dict) -> Dict:
        pc_noisy = batch["pc_noisy"].reshape(-1, self.patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, self.patch_size, 3)
        main_loss, commitment_loss, pred_pc, pc_clean = self.get_supervised_loss_with_commitment(
            pc_noisy=pc_noisy,
            pc_clean=pc_clean,
        )
        return {"loss": main_loss + commitment_loss}

    def denoise_langevin_dynamics(self, pc_noisy):
        with jt.no_grad():
            pred_disp, _ = self.feature_nets(
                pc_noisy,
                calculate_commitment_losses=False,
            )
        return pc_noisy + pred_disp

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch["pc_noisy"]
        result = []
        for pc_noisy in pc_noisy_batch:
            pc_next = pc_noisy
            for _ in range(self.niters):
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=self.patch_size,
                    seed_k=self.seed_k,
                    seed_k_alpha=self.seed_k_alpha,
                )
            result.append({"pc_denoised": pc_next.detach().numpy().astype(np.float32)})
        return result

    def validation_step(self, batch: Dict) -> Dict:
        pc_noisy = batch["pc_noisy"].reshape(-1, self.patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, self.patch_size, 3)
        pred = self.denoise_langevin_dynamics(pc_noisy)
        loss = chamfer_distance_unit_sphere(pred, pc_clean)
        return {"loss": loss}

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        out = []
        for asset in batch:
            if not self.is_predict():
                if asset.meta is not None and "pc_noisy" in asset.meta:
                    out.append(
                        {
                            "pc_noisy": asset.meta["pc_noisy"],
                            "pc_clean": asset.meta["pc_clean"],
                        }
                    )
                else:
                    out.append(
                        {
                            "pc_noisy": asset.sampled_vertices_noisy,
                            "pc_clean": asset.sampled_vertices,
                        }
                    )
            else:
                item = {"pc_noisy": asset.sampled_vertices_noisy}
                if asset.sampled_vertices is not None:
                    item["pc_clean"] = asset.sampled_vertices
                out.append(item)
        return out
