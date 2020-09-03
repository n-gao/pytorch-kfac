import torch.nn as nn
from .fisher_block import FisherBlock
from .linear_block import FullyConnectedFisherBlock
from .conv_block import ConvFisherBlock
from .identity import Identity


def init_fisher_block(module: nn.Module, **kwargs) -> FisherBlock:
    if type(module) is nn.Linear:
        layer = FullyConnectedFisherBlock(module)
    elif type(module) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        layer = ConvFisherBlock(module)
    elif type(module) is FisherBlock:
        layer = module
    else:
        layer = Identity(module)
    layer.setup(**kwargs)
    return layer
