# path: models/__init__.py

from .unet      import UNet
from .swin_unet import SwinUNet

_MODEL_REGISTRY = {
    'unet':      UNet,
    'swin_unet': SwinUNet,
}

def get_model(name: str, **kwargs):
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model arch: {name!r}")
    return _MODEL_REGISTRY[name](**kwargs)
