import unittest

from torch import nn, tensor

from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock, Identity, ConvFisherBlock
from torch_kfac.layers.fisher_block_factory import FisherBlockFactory


class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(12, 8, 4)
        self.conv2 = nn.Conv2d(9, 7, 2)
        self.conv3 = nn.Conv3d(7, 5, 2)
        self.conv4 = nn.ConvTranspose1d(1, 2, 3)


class KfacOptimizerTest(unittest.TestCase):
    def test_constructor_when_simple_model_should_create_fisher_blocks(self):
        """Tests that the KFAC constructor correctly creates Linear and Convolutional blocks
        for the corresponding layers."""
        model = MockModel()
        optimizer = KFAC(model, 0.01, tensor(1e-2))
        # model.modules(): SimpleModule, Linear, ReLU, Conv1d, Conv2d, Conv3d, ConvTranspose1d.
        self.assertIsInstance(optimizer.blocks[0], Identity)
        self.assertIsInstance(optimizer.blocks[1], FullyConnectedFisherBlock)
        self.assertIsInstance(optimizer.blocks[2], Identity)
        self.assertIsInstance(optimizer.blocks[3], ConvFisherBlock)
        self.assertIsInstance(optimizer.blocks[4], ConvFisherBlock)
        self.assertIsInstance(optimizer.blocks[5], ConvFisherBlock)
        self.assertIsInstance(optimizer.blocks[6], Identity)

    def test_constructor_when_simple_model_with_custom_blocks_should_use_custom_block(self):
        """Tests that the KFAC constructor correctly uses a custom fisher block registered
        in the block factory."""
        model = MockModel()
        factory = FisherBlockFactory()
        factory.register_block(nn.ConvTranspose1d, ConvFisherBlock)
        optimizer = KFAC(model, 0.01, tensor(1e-2), block_factory=factory)
        self.assertIsInstance(optimizer.blocks[6], ConvFisherBlock)


if __name__ == '__main__':
    unittest.main()
