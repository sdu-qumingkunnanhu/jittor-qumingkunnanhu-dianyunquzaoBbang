from dataclasses import dataclass
from typing import List, Optional

from .asset import Asset
from .augment import Augment, get_augments
from .spec import ConfigSpec

@dataclass
class Transform(ConfigSpec): # a simple class to wrap augments
    
    augments: Optional[List[Augment]]=None
    
    @classmethod
    def parse(cls, **kwargs) -> 'Transform':
        cls.check_keys(kwargs)
        augments_config = kwargs.get('augments')
        
        d = {}
        if augments_config is not None:
            d['augments'] = get_augments(*augments_config)
        return Transform(**d)
    
    def apply(self, asset: Asset):
        if self.augments is not None:
            for augment in self.augments:
                augment.apply(asset)