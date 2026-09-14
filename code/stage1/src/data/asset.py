from dataclasses import dataclass

from numpy import ndarray
from typing import Dict, Optional

import numpy as np
import os

@dataclass
class Asset():
    path: Optional[str]=None # where is the asset loaded from
    
    cls: Optional[str]=None # cls
    
    vertices: Optional[ndarray]=None # shape (N, 3)
    
    faces: Optional[ndarray]=None # shape (F, 3)
    
    sampled_vertices: Optional[ndarray]=None
    
    sampled_vertices_noisy: Optional[ndarray]=None
    
    meta: Optional[Dict]=None
    
    def transform(self, trans: ndarray):
        """trans: 4x4 affine matrix"""
        def _apply(v: ndarray, trans: ndarray) -> ndarray:
            return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]
        
        if self.vertices is not None:
            self.vertices = _apply(self.vertices, trans)

class Exporter(): # a simple parser
    
    @classmethod
    def _safe_make_dir(cls, path: str):
        if os.path.dirname(path) == '':
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    @classmethod
    def export_obj(cls, vertices, path: str, precision: int=6):
        lines = []
        for v in vertices:
            lines.append(f'v {v[0]:.{precision}f} {v[2]:.{precision}f} {-v[1]:.{precision}f}\n')
        cls._safe_make_dir(path)
        f = open(path, "w")
        f.writelines(lines)
        f.close()