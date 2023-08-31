__all__ = ['build_model']

from utils import Registry

META_ARCHITECTURES = Registry()

def build_model(cfg, device=None):
    model = META_ARCHITECTURES[cfg["model"]](cfg)
    return model.to(device)