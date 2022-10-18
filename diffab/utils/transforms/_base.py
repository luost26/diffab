import copy
import torch
from torchvision.transforms import Compose


_TRANSFORM_DICT = {}


def register_transform(name):
    def decorator(cls):
        _TRANSFORM_DICT[name] = cls
        return cls
    return decorator


def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = _TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))
    return Compose(tfms)


def _index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v


def _index_select_data(data, index):
    return {
        k: _index_select(v, index, data['aa'].size(0))
        for k, v in data.items()
    }


def _mask_select(v, mask):
    if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
        return v[mask]
    elif isinstance(v, list) and len(v) == mask.size(0):
        return [v[i] for i, b in enumerate(mask) if b]
    else:
        return v


def _mask_select_data(data, mask):
    return {
        k: _mask_select(v, mask)
        for k, v in data.items()
    }
