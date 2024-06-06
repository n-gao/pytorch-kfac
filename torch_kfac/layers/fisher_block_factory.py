from typing import Type, Tuple, Sequence

from torch import nn

from torch_kfac.layers import FisherBlock, FullyConnectedFisherBlock, Identity, ConvFisherBlock
from torch_kfac.layers.fisher_block import ExtensionFisherBlock

ModuleBlockList = Sequence[Tuple[Type[nn.Module], Type[ExtensionFisherBlock]]]

default_blocks: ModuleBlockList = (
    (nn.Linear, FullyConnectedFisherBlock),
    (nn.Conv1d, ConvFisherBlock),
    (nn.Conv2d, ConvFisherBlock),
    (nn.Conv3d, ConvFisherBlock),
)


class FisherBlockFactory:
    """
    Factory for subclasses of FisherBlock. Can be used to create the
    correct fisher block for each torch module.
    """

    def __init__(self, blocks_to_register: ModuleBlockList = default_blocks):
        """
        Initializes the fisher block factory.
        Args:
            blocks_to_register: The block types which should be registered. By default, registers blocks for
                Linear and Convolutional layers.
        """
        self.blocks: ModuleBlockList = []

        self.register_blocks(blocks_to_register)

    def register_blocks(self, block_list: ModuleBlockList):
        for module_type, block_type in block_list:
            self.register_block(module_type, block_type)

    def register_block(self, module_type: Type[nn.Module], block_type: Type[ExtensionFisherBlock]):
        """
        Registers a new fisher block, which is able to handle the specified torch module type.
        Args:
            module_type: The torch module type for which this block can be used.
            block_type: The type of the block.
        """
        self.blocks.append((module_type, block_type))

    def create_block(self, module: nn.Module, **kwargs) -> FisherBlock:
        """
        Creates and sets up a new fisher block, which is able to handle the specified torch module type.
        If no block is found, then this method returns the Identity block, which does not
        apply preconditioning.
        Args:
            module: an instance of a torch module (e.g. nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d).
            **kwargs: the keyword arguments which are passed to the setup method of the created block.

        Returns:
            The initialized fisher block for the specified torch module.
        """
        module_type = type(module)
        block = Identity(module)
        for block_module_type, block_type in self.blocks:
            if block_module_type is module_type:
                block = block_type(module)
                break
        block.setup(**kwargs)
        return block
