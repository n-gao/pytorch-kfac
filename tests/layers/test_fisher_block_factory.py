import unittest
from unittest.mock import patch

import torch.nn
from torch import nn

from torch_kfac.layers import Identity, FisherBlock
from torch_kfac.layers.fisher_block import ExtensionFisherBlock
from torch_kfac.layers.fisher_block_factory import FisherBlockFactory, default_blocks
from torch_kfac.utils import Lock


class MockFisherBlock1(ExtensionFisherBlock):
    def __init__(self, module: torch.nn.Module, **kwargs) -> None:
        super().__init__(
            module=module,
            in_features=module.in_features + int(module.bias is not None),
            out_features=module.out_features,
            dtype=module.weight.dtype,
            device=module.weight.device,
            **kwargs)


class MockFisherBlock2(MockFisherBlock1):
    pass


class SubLinearModule(torch.nn.Linear):
    pass


class FisherBlockFactoryTest(unittest.TestCase):

    def setUp(self):
        self.module = nn.Linear(2, 2)
        self.forward_lock = Lock()
        self.backward_lock = Lock()
        self.center = False
        self.enable_pi_correction = True
        self.kwargs = { "center": self.center, "forward_lock": self.forward_lock,
                        "enable_pi_correction": self.enable_pi_correction,
                        "backward_lock": self.backward_lock }

    def test_constructor_when_register_default_blocks_unspecified_should_add_blocks(self):
        """Tests that the default blocks are added."""
        factory = FisherBlockFactory()
        self.assertEqual(len(default_blocks), len(factory.blocks))

    def test_constructor_when_register_default_blocks_false_should_add_blocks(self):
        """Tests that the default blocks are not added if register_default_blocks is set to false."""
        factory = FisherBlockFactory([])
        self.assertEqual(0, len(factory.blocks))

    def test_create_block_when_no_blocks_available_should_return_identity(self):
        """Tests if the identity block is returned by create_block when no blocks are available."""
        factory = FisherBlockFactory([])
        block = factory.create_block(self.module, **self.kwargs)
        self.assertIsInstance(block, Identity)

    def test_create_block_check_passed_arguments_to_setup(self):
        """Tests that create_block passed the correct arguments to the setup method."""
        factory = FisherBlockFactory([])
        with patch.object(Identity, 'setup', return_value=None) as mock_method:
            factory.create_block(self.module, **self.kwargs)
        mock_method.assert_called_once_with(**self.kwargs)

    def test_create_block_when_suitable_block_registered_should_return_correct_block(self):
        """Tests that create_block selects the correct fisher block for a linear module."""
        factory = FisherBlockFactory([])
        factory.register_block(nn.AdaptiveMaxPool2d, MockFisherBlock2)
        factory.register_block(nn.Linear, MockFisherBlock1)
        block = factory.create_block(self.module, **self.kwargs)
        self.assertIsInstance(block, MockFisherBlock1)

    def test_create_block_when_two_blocks_registered_for_same_module_should_return_correct_block(self):
        """Tests that the first registered fisher block is used if two blocks are
        registered for the same module type."""
        factory = FisherBlockFactory([])
        factory.register_block(nn.Linear, MockFisherBlock2)
        factory.register_block(nn.Linear, MockFisherBlock1)
        block = factory.create_block(self.module, **self.kwargs)
        self.assertIsInstance(block, MockFisherBlock2)

    def test_create_block_when_subclass_module_block_registered_should_return_correct_block(self):
        """
        Tests that if create_block is called with a submodule, which inherits from nn.Linear,
        uses the block which is specifically for this submodule and not the generic one
        for nn.Linear.
        """
        factory = FisherBlockFactory([])
        factory.register_block(nn.Linear, MockFisherBlock1)
        factory.register_block(SubLinearModule, MockFisherBlock2)
        module = SubLinearModule(2, 2)
        block = factory.create_block(module, **self.kwargs)
        self.assertIsInstance(block, MockFisherBlock2)


if __name__ == '__main__':
    unittest.main()
