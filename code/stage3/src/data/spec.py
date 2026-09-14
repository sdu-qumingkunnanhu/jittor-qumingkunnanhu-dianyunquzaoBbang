from abc import ABC, abstractmethod
from dataclasses import fields

class ConfigSpec(ABC):
    @classmethod
    # 检查配置字段有没有写错
    def check_keys(cls, config, expect=None):
        if expect is None:
            expect = [field.name for field in fields(cls)] # type: ignore
        for key in config.keys():
            # 如果某个字段不是配置类允许的字段，就认为配置写错了
            if key not in expect:
                raise ValueError(f"expect names {expect} in {cls.__name__}, found {key}")
    
    @classmethod
    @abstractmethod
    # 规定所有继承ConfigSpec的配置类都应该实现
    def parse(cls, **kwargs) -> 'ConfigSpec':
        raise NotImplementedError()