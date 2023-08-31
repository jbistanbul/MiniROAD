__all__ = [
    'build_criterion'
]

from utils import Registry

CRITERIONS = Registry()

def build_criterion(cfg, device=None):
    criterion = CRITERIONS[cfg["loss"]](cfg)
    return criterion.to(device)