__all__ = [
    'build_trainer'
]

from utils import Registry

TRAINER = Registry()

def build_trainer(cfg):
    trainer = TRAINER[cfg["task"]]
    return trainer
