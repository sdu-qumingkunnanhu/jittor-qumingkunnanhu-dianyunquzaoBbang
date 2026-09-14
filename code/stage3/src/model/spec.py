from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import jittor as jt
from jittor import nn
import numpy as np

from ..data.asset import Asset
from ..data.transform import Transform


@dataclass
class ModelInput:
    asset: Asset
    tokens: Optional[np.ndarray] = None


class ModelSpec(nn.Module, ABC):
    def __init__(self, model_config, transform_config):
        super().__init__()
        self.model_config = dict(model_config)
        self.transform_config = dict(transform_config)
        self._is_predict = False

    def is_predict(self):
        return self._is_predict

    def set_predict(self, is_predict: bool):
        self._is_predict = is_predict

    def _process_fn(self, batch: List[Asset]) -> List[Dict]:
        processed = self.process_fn(batch)
        if not self.is_training():
            for i, asset in enumerate(batch):
                non = processed[i].get("non", {})
                non["asset"] = deepcopy(asset)
                processed[i]["non"] = non
        return processed

    @abstractmethod
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        raise NotImplementedError()

    def get_train_transform(self) -> Optional[Transform]:
        cfg = self.transform_config.get("train_transform")
        return None if cfg is None else Transform.parse(**cfg)

    def get_validate_transform(self) -> Optional[Transform]:
        cfg = self.transform_config.get("validate_transform")
        return None if cfg is None else Transform.parse(**cfg)

    def get_predict_transform(self) -> Optional[Transform]:
        cfg = self.transform_config.get("predict_transform")
        return None if cfg is None else Transform.parse(**cfg)

    def predict_step(self, batch: Dict) -> List[Dict]:
        raise NotImplementedError()

