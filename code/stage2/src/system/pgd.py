from typing import Dict, List, Optional

import numpy as np
import os

from .spec import DummySystem, DummyWriter
from .spec import resolve_project_output
from ..data.asset import Exporter


class PGDWriter(DummyWriter):
    def __init__(self, save_dir: str = "predictions/pgd", save_name: str = "denoised", output_format: str = "npy"):
        self.save_dir = resolve_project_output(save_dir)
        self.save_name = save_name
        self.output_format = output_format
        assert self.output_format in {"npy", "obj"}, "output_format must be npy or obj"

    def write(self, batch, prediction: List[Dict], dataset_module=None):
        for i, asset in enumerate(batch["asset"]):
            assert asset.path is not None, "asset path is None"
            relpath = os.path.basename(asset.path)
            rel_dir = os.path.dirname(relpath)
            out_dir = os.path.join(self.save_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            denoised = prediction[i]["pc_denoised"]
            if not isinstance(denoised, np.ndarray):
                denoised = denoised.numpy()
            denoised = denoised.astype(np.float32)

            if self.output_format == "npy":
                np.save(os.path.join(out_dir, f"{self.save_name}.npy"), denoised)
            else:
                Exporter.export_obj(denoised, os.path.join(out_dir, f"{self.save_name}.obj"))


class PGDSystem(DummySystem):
    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter] = None,
        ckpt_save_dir: str = "experiments/pgd",
        ckpt_save_name: str = "checkpoint_pgd",
    ):
        super().__init__(
            dataset_module=dataset_module,
            model=model,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            writer=writer,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_save_name=ckpt_save_name,
        )
