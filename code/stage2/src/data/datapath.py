from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass
from random import shuffle
from typing import Dict, List, Optional

import numpy as np
import os

from .asset import Asset
from .spec import ConfigSpec


_OBJ_MESH_CACHE: Dict[str, tuple[np.ndarray, np.ndarray]] = {}

@dataclass
class LazyAsset(ABC):
    """store datapath and load upon requiring"""
    path: str
    
    cls: Optional[str]=None
    
    @abstractmethod
    def load(self) -> 'Asset':
        raise NotImplementedError()

@dataclass
class ObjLazyAsset(LazyAsset):
    # load an obj file as an asset
    def load(self) -> 'Asset':
        cached = _OBJ_MESH_CACHE.get(self.path)
        if cached is None:
            import trimesh
            mesh = trimesh.load(self.path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            cached = (
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
            )
            _OBJ_MESH_CACHE[self.path] = cached
        vertices, faces = cached
        asset = Asset(
            path=self.path,
            cls=self.cls,
            vertices=vertices.copy(),
            faces=faces.copy(),
        )
        return asset

@dataclass
class NpyLazyAsset(LazyAsset):
    # load a .npy point cloud file as an asset (for predict mode)
    def load(self) -> 'Asset':
        pc = np.load(self.path).astype(np.float64)
        asset = Asset(
            path=self.path,
            cls=self.cls,
            sampled_vertices_noisy=pc,
        )
        return asset

@dataclass
class Datapath(ConfigSpec):
    """handle input data paths"""
    
    # all filepaths
    filepaths: List[str]
    
    # root to add to prefix
    input_dataset_dir: str=''
    
    # name of class
    cls_name: Optional[List[str]]=None
    
    # bias in a single class
    cls_bias: Optional[List[int]]=None
    
    # num of files in a single class
    cls_length: Optional[List[int]]=None
    
    # how many files to return when using data sampling
    num_files: Optional[int]=None
    
    # use proportion data sampling
    use_prob: bool=False
    
    # weight
    cls_weight: Optional[List[float]]=None
    
    # use bpy loader
    loader: type[LazyAsset]=ObjLazyAsset
    
    # data name
    data_name: Optional[str]=None
    
    # check if path exists
    ignore_check: bool=False
    
    @classmethod
    def parse(cls, **kwargs) -> 'Datapath':
        MAP = {
            None: ObjLazyAsset,
            'obj': ObjLazyAsset,
            'npy': NpyLazyAsset,
        }
        input_dataset_dir = kwargs.get('input_dataset_dir', '')
        num_files = kwargs.get('num_files', None)
        use_prob = kwargs.get('use_prob', False)
        data_name = kwargs.get('data_name', 'raw_data.npz')
        data_path = kwargs.get('data_path', None)
        loader_cls = MAP[kwargs.get('loader', None)]
        ignore_check = kwargs.get('ignore_check', False)
        
        if data_path is not None:
            filepaths = []
            if isinstance(data_path, dict):
                cls_name = []
                cls_bias = []
                cls_length = []
                cls_weight = []
                for name, v in data_path.items():
                    assert isinstance(v, list), "items in the dict must be a list of data list paths"
                    for item in v:
                        if isinstance(item, str):
                            datalist_path = item
                            weight = 1.0
                        else:
                            datalist_path = item[0]
                            weight = item[1]
                        cls_name.append(name)
                        lines = [x.strip() for x in open(datalist_path, "r").readlines()]
                        ok_lines = []
                        missing = 0
                        for line in lines:
                            if ignore_check:
                                ok_lines.append(line)
                            elif os.path.exists(os.path.join(input_dataset_dir, line, data_name)):
                                ok_lines.append(line)
                            else:
                                missing += 1
                        if missing != 0:
                            print(f"\033[31m{datalist_path}: {missing} missing files\033[0m")
                        cls_bias.append(len(filepaths))
                        cls_length.append(len(ok_lines))
                        cls_weight.append(weight)
                        filepaths.extend(ok_lines)
            else:
                raise NotImplementedError()
        else:
            _filepaths = kwargs['filepaths']
            if isinstance(_filepaths, list):
                filepaths = _filepaths
                cls_name = None
                cls_bias = None
                cls_length = None
                cls_weight = None
            elif isinstance(_filepaths, dict):
                filepaths = []
                cls_name = []
                cls_bias = []
                cls_length = []
                cls_weight = []
                for k, v in _filepaths.items():
                    assert isinstance(v, list), "items in the dict must be a list of paths"
                    cls_name.append(k)
                    cls_bias.append(len(filepaths))
                    cls_length.append(len(v))
                    cls_weight.append(1.0)
                    filepaths.extend(v)
            else:
                raise NotImplementedError()
        if cls_weight is not None:
            total = sum(cls_weight)
            cls_weight = [x/total for x in cls_weight]
        return Datapath(
            filepaths=filepaths,
            input_dataset_dir=input_dataset_dir,
            cls_name=cls_name,
            cls_bias=cls_bias,
            cls_length=cls_length,
            num_files=num_files,
            use_prob=use_prob,
            cls_weight=cls_weight,
            loader=loader_cls,
            data_name=data_name,
            ignore_check=ignore_check,
        )
    
    def make(self, path: str, cls: Optional[str]) -> LazyAsset:
        return self.loader(path=path, cls=cls)
    
    def __getitem__(self, index: int) -> LazyAsset:
        if self.use_prob and self.cls_weight is not None:
            if self.cls_bias is None:
                raise ValueError("do not have cls_bias")
            if self.cls_length is None:
                raise ValueError("do not have cls_length")
            if not hasattr(self, "perms"):
                self.perms = []
                self.current_bias = []
                for i in range(len(self.cls_weight)):
                    self.perms.append([x for x in range(self.cls_length[i])])
                    self.current_bias.append(0)
            idx = np.random.choice(len(self.cls_weight), p=self.cls_weight)
            i = self.perms[idx][self.current_bias[idx]]
            self.current_bias[idx] += 1
            if self.current_bias[idx] >= self.cls_length[idx]:
                shuffle(self.perms[idx])
                self.current_bias[idx] = 0
            if self.cls_name is None:
                name = None
            else:
                name = self.cls_name[idx]
            path = os.path.join(self.input_dataset_dir, self.filepaths[i+self.cls_bias[idx]])
            if self.data_name is not None:
                path = os.path.join(path, self.data_name)
            return self.make(path=path, cls=name)
        else:
            if self.cls_name is None or self.cls_bias is None or self.cls_length is None:
                name = None
            else:
                name = None
                for i in range(len(self.cls_bias)):
                    start = self.cls_bias[i]
                    end = start + self.cls_length[i]
                    if start <= index < end:
                        name = self.cls_name[i]
                        break
            path = os.path.join(self.input_dataset_dir, self.filepaths[index])
            if self.data_name is not None:
                path = os.path.join(path, self.data_name)
            return self.make(path=path, cls=name)
    
    def get_data(self) -> List[LazyAsset]:
        return [self[i] for i in range(len(self))]
    
    def split_by_cls(self) -> Dict[Optional[str], 'Datapath']:
        res: Dict[Optional[str], Datapath] = {}
        if self.cls_name is None:
            res[None] = self
            return res
        if self.cls_bias is None:
            raise ValueError("do not have cls_bias")
        if self.cls_length is None:
            raise ValueError("do not have cls_length")
        d_filepaths = defaultdict(list)
        d_length = defaultdict(int)
        d_weight = defaultdict(list)
        for (i, cls) in enumerate(self.cls_name):
            s = slice(self.cls_bias[i], self.cls_bias[i]+self.cls_length[i])
            d_filepaths[cls].extend(self.filepaths[s].copy())
            d_length[cls] += self.cls_length[i]
            if self.cls_weight is not None:
                d_weight[cls].append(self.cls_weight[i])
        for cls in d_filepaths:
            cls_weight = None if self.cls_weight is None else d_weight[cls]
            if cls_weight is not None:
                total = sum(cls_weight)
                cls_weight = [x/total for x in cls_weight]
            res[cls] = Datapath(
                filepaths=d_filepaths[cls],
                input_dataset_dir=self.input_dataset_dir,
                cls_name=[cls],
                cls_bias=[0],
                cls_length=[len(d_filepaths[cls])],
                num_files=self.num_files,
                use_prob=self.use_prob,
                cls_weight=cls_weight,
                loader=self.loader,
                data_name=self.data_name,
            )
        return res
    
    def __len__(self):
        if self.use_prob:
            assert self.num_files is not None, 'num_files is not specified'
            return self.num_files
        return len(self.filepaths)
