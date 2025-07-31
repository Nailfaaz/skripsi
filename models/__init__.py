# path: models/__init__.py

from .unet         import UNet
from .swin_unet    import SwinUNet
from .mobile_unetr import MobileUNetR

_MODEL_REGISTRY = {
    'unet':        UNet,
    'swin_unet':   SwinUNet,
    'mobile_unetr':MobileUNetR,
}

def get_model(name: str, **kwargs):
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown arch: {name}")
    return _MODEL_REGISTRY[name](**kwargs)
