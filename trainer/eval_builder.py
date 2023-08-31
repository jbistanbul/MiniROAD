__all__ = [
    'build_eval'
]

from utils import Registry

EVAL = Registry()

def build_eval(cfg):
    eval = EVAL[cfg["task"]](cfg)
    return eval
