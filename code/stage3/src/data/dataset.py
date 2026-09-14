from dataclasses import dataclass
from jittor import dataset
from jittor.dataset import Dataset
from numpy import ndarray
from typing import List, Dict, Callable, Optional, Union

import jittor as jt
import numpy as np
import os
import random

from .asset import Asset
from .augment import Augment
from .datapath import Datapath, LazyAsset
from .spec import ConfigSpec
from .transform import Transform


def _stable_int(value) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value & 0xFFFFFFFF
    data = str(value).encode("utf-8")
    out = 2166136261
    for b in data:
        out ^= b
        out = (out * 16777619) & 0xFFFFFFFF
    return out


def _mix_seed(*values) -> int:
    seed = 0x6D2B79F5
    for value in values:
        v = _stable_int(value)
        seed ^= v + 0x9E3779B9 + ((seed << 6) & 0xFFFFFFFF) + (seed >> 2)
        seed &= 0xFFFFFFFF
    return seed


def _seed_numpy_python(seed: Optional[int]):
    if seed is None:
        return
    seed = int(seed) & 0xFFFFFFFF
    np.random.seed(seed)
    random.seed(seed)


_PATH_EXISTS_CACHE: Dict[str, bool] = {}


def _path_exists(path: str) -> bool:
    cached = _PATH_EXISTS_CACHE.get(path)
    if cached is None:
        cached = os.path.isfile(path)
        _PATH_EXISTS_CACHE[path] = cached
    return cached


@dataclass
class DatasetConfig(ConfigSpec):
    shuffle: bool
    batch_size: int
    num_workers: int
    datapath: Datapath
    drop_last: bool = False
    rebuild_each_epoch: bool = False
    
    @classmethod
    def parse(cls, **kwargs) -> 'DatasetConfig':
        cls.check_keys(kwargs)
        return DatasetConfig(
            shuffle=kwargs.get('shuffle', False),
            batch_size=kwargs.get('batch_size', 1),
            num_workers=kwargs.get('num_workers', 0),
            datapath=Datapath.parse(**kwargs.get('datapath')), # type: ignore
            drop_last=kwargs.get('drop_last', False),
            rebuild_each_epoch=kwargs.get('rebuild_each_epoch', False),
        )
    
    # 把一个包含多个类别的数据集的配置，拆成每个类别一个独立的DatasetConfig
    def split_by_cls(self) -> Dict[Optional[str], 'DatasetConfig']: # 返回一个字典
        # 创建空字典，用于保存拆分结果 
        res: Dict[Optional[str], DatasetConfig] = {}
        # 先让内部的Datapath自己按类别拆分
        datapath_dict = self.datapath.split_by_cls()
        # 逐个遍历每个类别
        for cls, v in datapath_dict.items():
            res[cls] = DatasetConfig(
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                datapath=v,
                drop_last=self.drop_last,
                rebuild_each_epoch=self.rebuild_each_epoch,
            )
        return res

class PCDatasetModule():
    def __init__(
        self,
        # 模型提供的数据整理函数
        process_fn: Optional[Callable[[List[Asset]], List[Dict]]]=None,
        # 训练集配置
        train_dataset_config: Optional[DatasetConfig]=None,
        # 验证集配置
        validate_dataset_config: Optional[Dict[Optional[str], DatasetConfig]]=None,
        # 预测集配置
        predict_dataset_config: Optional[Dict[Optional[str], DatasetConfig]]=None,
        # 训练阶段的数据变换
        train_transform: Optional[Transform]=None,
        # 验证阶段的数据变换
        validate_transform: Optional[Transform]=None,
        # 正式预测阶段的数据变换
        predict_transform: Optional[Transform]=None,
        # 调试模式开关
        debug: bool=False,
        # 主程序传入的随机种子；None 时保持原始行为
        seed: Optional[int]=None,
    ):
        self.process_fn                 = process_fn
        self.train_dataset_config       = train_dataset_config
        self.validate_dataset_config    = validate_dataset_config
        self.predict_dataset_config     = predict_dataset_config
        self.train_transform            = train_transform
        self.validate_transform         = validate_transform
        self.predict_transform          = predict_transform
        self.debug = debug
        self.seed = None if seed is None else int(seed)
        self._split_counters = {
            "train": 0,
            "validate": 0,
            "predict": 0,
        }
        
        if debug:
            print("\033[31mWARNING: debug mode\033[0m")
        
        # build train datapath
        if self.train_dataset_config is not None:
            self.train_datapath = self.train_dataset_config.datapath
        else:
            self.train_datapath = None
        
        # build validate datapath
        if self.validate_dataset_config is not None:
            self.validate_datapath = {
                cls: self.validate_dataset_config[cls].datapath
                for cls in self.validate_dataset_config
            }
        else:
            self.validate_datapath = None
        
        # build predict datapath
        if self.predict_dataset_config is not None:
            self.predict_datapath = {
                cls: self.predict_dataset_config[cls].datapath
                for cls in self.predict_dataset_config
            }
        else:
            self.predict_datapath = None

    def _next_seed(self, split: str, cls: Optional[str]=None) -> Optional[int]:
        if self.seed is None:
            return None
        counter = self._split_counters[split]
        self._split_counters[split] = counter + 1
        return _mix_seed(self.seed, split, counter, cls)
        
    def train_dataloader(self):
        if self.train_transform is not None and self.train_dataset_config is not None and self.train_datapath is not None:
            random_seed = self._next_seed("train")
            _seed_numpy_python(random_seed)
            if self.train_dataset_config.rebuild_each_epoch or not hasattr(self, "_train_ds"):
                self._train_ds = PCDataset(
                    data=self.train_datapath.get_data(),
                    transform=self.train_transform,
                    name="train",
                    process_fn=self.process_fn,
                    debug=self.debug,
                    random_seed=random_seed,
                )
            elif self.train_datapath.use_prob:
                # `use_prob` is stateful in Datapath: each call to get_data()
                # advances its sampling cursor. Update the data in place so the
                # sampled subset changes each epoch without rebuilding Jittor's
                # worker-backed Dataset object.
                self._train_ds.set_data(self.train_datapath.get_data())
                self._train_ds.random_seed = random_seed
            else:
                self._train_ds.random_seed = random_seed
        else:
            return None
        return self._create_dataloader(
            dataset=self._train_ds,
            config=self.train_dataset_config,
        )

    def validate_dataloader(self):
        if self.validate_dataset_config is not None and self.validate_transform is not None and self.validate_datapath is not None:
            self._validation_ds = {}
            for cls in self.validate_datapath:
                random_seed = self._next_seed("validate", cls)
                _seed_numpy_python(random_seed)
                self._validation_ds[cls] = PCDataset(
                    data=self.validate_datapath[cls].get_data(),
                    transform=self.validate_transform,
                    name=f"validate-{cls}",
                    process_fn=self.process_fn,
                    debug=self.debug,
                    random_seed=random_seed,
                )
        else:
            return None
        return self._create_dataloader(
            dataset=self._validation_ds,
            config=self.validate_dataset_config,
        )
    
    def predict_dataloader(self):
        if self.predict_transform is not None and self.predict_dataset_config is not None and self.predict_datapath is not None:
            self._predict_ds = {}
            for cls in self.predict_datapath:
                random_seed = self._next_seed("predict", cls)
                _seed_numpy_python(random_seed)
                self._predict_ds[cls] = PCDataset(
                    data=self.predict_datapath[cls].get_data(),
                    transform=self.predict_transform,
                    name=f"predict-{cls}",
                    process_fn=self.process_fn,
                    debug=self.debug,
                    random_seed=random_seed,
                )
        else:
            return None
        return self._create_dataloader(
            dataset=self._predict_ds,
            config=self.predict_dataset_config,
        )

    def _create_dataloader(
        self,
        dataset: Union[Dataset, Dict[str, Dataset]],
        config: Union[DatasetConfig, Dict[Optional[str], DatasetConfig]],
        **kwargs,
    ) -> Union[Dataset, Dict[str, Dataset]]:
        def create_single_dataloader(dataset: Dataset, config: DatasetConfig, **kwargs):
            random_seed = getattr(dataset, "random_seed", None)
            if random_seed is not None and hasattr(dataset, "_shuffle_rng"):
                dataset._shuffle_rng = np.random.default_rng(int(random_seed) & 0xFFFFFFFF)
            if not getattr(dataset, "_pc_attrs_initialized", False):
                dataset.set_attrs(
                    batch_size=config.batch_size,
                    total_len=len(dataset),# len(config.datapath),
                    shuffle=config.shuffle,
                    num_workers=config.num_workers,
                    drop_last=config.drop_last,
                )
                dataset._pc_attrs_initialized = True
            return dataset
        if isinstance(dataset, Dict):
            assert isinstance(config, dict)
            return {k: create_single_dataloader(v, config[k], **kwargs) for k, v in dataset.items()}
        else:
            assert isinstance(config, DatasetConfig)
            return create_single_dataloader(dataset, config, **kwargs)


class PCDataset(Dataset):
    '''
    A simple dataset class.
    '''
    def __init__(
        self,
        data: List[LazyAsset],
        transform: Transform,
        name: Optional[str]=None,
        process_fn: Optional[Callable[[List[Asset]], List[Dict]]]=None,
        debug: bool=False,
        random_seed: Optional[int]=None,
    ):
        super().__init__()

        self.name       = name
        self.process_fn = process_fn
        self.transform  = transform
        self.debug      = debug
        self.random_seed = None if random_seed is None else int(random_seed)
        self.set_data(data)
        
        if not debug:
            assert self.process_fn is not None, 'missing data processing function'

    def set_data(self, data: List[LazyAsset]):
        # 用于过滤不存在的文件
        existing_data = []
        missing_count = 0

        for x in data:
            path = getattr(x, "path", None)
            if path is None:
                existing_data.append(x)
            elif _path_exists(path):
                existing_data.append(x)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"[Dataset] Skip {missing_count} missing files in {self.name}")

        self.data = existing_data
    
    def __len__(self) -> int:
        return len(self.data)

    def _seed_for_index(self, index):
        if self.random_seed is None:
            return
        _seed_numpy_python(_mix_seed(self.random_seed, int(index)))
    
    def __getitem__(self, index) -> Asset:
        self._seed_for_index(index)
        # 加载数据
        # 训练集提供三维物体的原始干净网格文件（.obj 格式），每个文件包含顶点和面片信息。
        lazy_asset = self.data[index]
        asset = lazy_asset.load()
        # main
        self.transform.apply(asset=asset)
        return asset
    
    def _collate_fn_debug(self, batch):
        return batch # just retun a list of Asset
    
    def _collate_fn(self, batch):
        processed_batch = self.process_fn(batch) # type: ignore
        processed_batch: List[Dict]
        
        tensors_stack = {}
        tensors_cat = {}
        non_tensors = {}
        vis = {}
        def check(x):
            assert x not in vis, f"multiple keys found: {x}"
            vis[x] = True
        
        for k, v in processed_batch[0].items():
            if k == "cat":
                assert isinstance(v, dict)
                for k1 in v.keys():
                    check(k1)
                    tensors_cat[k1] = []
                    for i in range(len(processed_batch)):
                        v1 = processed_batch[i]['cat'][k1]
                        if isinstance(v1, ndarray):
                            v1 = jt.array(v1)
                        elif isinstance(v1, jt.Var):
                            v1 = v1
                        else:
                            raise ValueError(f"cannot concatenate non-tensor type of key {k1}, type: {type(v1)}")
                        tensors_cat[k1].append(v1)
            elif k == "non":
                assert isinstance(v, dict)
                for k1 in v.keys():
                    check(k1)
                    non_tensors[k1] = []
                    for i in range(len(processed_batch)):
                        v1 = processed_batch[i]['non'][k1]
                        if isinstance(v1, ndarray):
                            v1 = jt.array(v1)
                        non_tensors[k1].append(v1)
            else:
                check(k)
                tensors_stack[k] = []
                for i in range(len(processed_batch)):
                    v1 = processed_batch[i][k]
                    if isinstance(v1, ndarray):
                        v1 = jt.array(v1)
                    elif isinstance(v1, jt.Var):
                        v1 = v1
                    else:
                        raise ValueError(f"cannot stack type of key {k}, type: {type(v1)}")
                    tensors_stack[k].append(v1)
        
        collated_stack = {k: jt.stack(v) for k, v in tensors_stack.items()}
        collated_cat = {k: jt.concat(v, dim=1) for k, v in tensors_cat.items()}
        
        collated_batch = {
            **collated_stack,
            **collated_cat,
            **non_tensors,
        }
        return collated_batch
    
    def collate_batch(self, batch):
        if self.debug:
            return self._collate_fn_debug(batch)
        return self._collate_fn(batch)
