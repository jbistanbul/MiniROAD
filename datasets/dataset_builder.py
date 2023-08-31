__all__ = [
    'build_dataset',
    'build_data_loader',
]

import torch.utils.data as data
from utils import Registry

DATA_LAYERS = Registry()

def build_dataset(cfg):
    data_layer = DATA_LAYERS[f'{cfg["data_name"]}']
    return data_layer

def build_data_loader(cfg, mode):
    data_layer = build_dataset(cfg)
    data_loader = data.DataLoader(
        dataset=data_layer(cfg, mode),
        batch_size=cfg["batch_size"] if mode == 'train' else cfg["test_batch_size"],
        shuffle=True if mode == 'train' else False,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )
    return data_loader