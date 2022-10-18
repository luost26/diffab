from torch.utils.data import Dataset, ConcatDataset
from diffab.utils.transforms import get_transform


_DATASET_DICT = {}


def register_dataset(name):
    def decorator(cls):
        _DATASET_DICT[name] = cls
        return cls
    return decorator


def get_dataset(cfg):
    transform = get_transform(cfg.transform) if 'transform' in cfg else None
    return _DATASET_DICT[cfg.type](cfg, transform=transform)


@register_dataset('concat')
def get_concat_dataset(cfg):
    datasets = [get_dataset(d) for d in cfg.datasets]
    return ConcatDataset(datasets)


@register_dataset('balanced_concat')
class BalancedConcatDataset(Dataset):

    def __init__(self, cfg, transform=None):
        super().__init__()
        assert transform is None, 'transform is not supported.'
        self.datasets = [get_dataset(d) for d in cfg.datasets]
        self.max_size = max([len(d) for d in self.datasets])

    def __len__(self):
        return self.max_size * len(self.datasets)

    def __getitem__(self, idx):
        dataset_idx = idx // self.max_size
        return self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])]
