import torch.nn as nn
from .layer import Layer
from .linear_layer import LinearLayer
from .conv_layer import ConvLayer
from .identity import IdentityLayer


def init_layer(module: nn.Module, **kwargs) -> Layer:
    if type(module) is nn.Linear:
        layer = LinearLayer(module)
    elif type(module) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        layer = ConvLayer(module)
    elif type(module) is Layer:
        layer = module
    else:
        layer = IdentityLayer(module)
    layer.setup(**kwargs)
    return layer
