from .pgd import PGDSystem, PGDWriter
from .spec import DummySystem, DummyWriter


def get_system(**kwargs) -> DummySystem:
    mapping = {
        "dummy": DummySystem,
        "pgd": PGDSystem,
    }
    target = kwargs["__target__"]
    assert target in mapping, f"expect: [{','.join(mapping.keys())}], found: {target}"
    cfg = dict(kwargs)
    del cfg["__target__"]
    return mapping[target](**cfg)


def get_writer(**kwargs) -> DummyWriter:
    mapping = {
        "dummy": DummyWriter,
        "pgd": PGDWriter,
    }
    target = kwargs["__target__"]
    assert target in mapping, f"expect: [{','.join(mapping.keys())}], found: {target}"
    cfg = dict(kwargs)
    del cfg["__target__"]
    return mapping[target](**cfg)
