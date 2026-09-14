from .pgd import PGDModel
from .spec import ModelSpec


def get_model(model_config, **kwargs) -> ModelSpec:
    mapping = {
        "PGDModel": PGDModel,
    }
    target = model_config["__target__"]
    cfg = dict(model_config)
    del cfg["__target__"]
    assert target in mapping, f"expect: [{','.join(mapping.keys())}], found: {target}"
    return mapping[target](model_config=cfg, **kwargs)
