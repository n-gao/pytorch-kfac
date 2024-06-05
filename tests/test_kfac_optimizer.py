import unittest
from unittest.mock import MagicMock

from torch import tensor, float64
from torch.nn import Linear

from torch_kfac import KFAC
from ddt import ddt, data

@ddt
class KfacOptimizerTest(unittest.TestCase):

    def setUp(self):
        self.model = Linear(2, 3, dtype=float64)
        self.input = tensor([[1, 0], [0, 1], [0.5, 0.5], [2, 0.5], [4, 1.1]], dtype=float64)
        self.damping = tensor(1e-1)
        self.preconditioner = KFAC(self.model, 0, self.damping, update_cov_manually=True)
        self.test_block = self.preconditioner.blocks[0]

    def forward_backward_pass(self):
        with self.preconditioner.track_forward():
            loss = self.model(self.input).norm()
        with self.preconditioner.track_backward():
            loss.backward()

    @data(True, False)
    def test_update_cov_should_update_inverses(self, update_inverses: bool):
        self.forward_backward_pass()
        self.preconditioner.blocks[0].update_cov_inv = MagicMock()
        self.preconditioner.update_cov(update_inverses)
        self.assertEqual(update_inverses, self.preconditioner.blocks[0].update_cov_inv.called)

    def test_disable_pi_correction(self):
        """
        Checks if the enable_pi_correction parameter is correctly copied to the Fisher blocks.
        """
        model = Linear(3, 4)
        preconditioner = KFAC(model, 0.01, tensor(1e-2), enable_pi_correction=False)
        self.assertEqual(1, len(preconditioner.blocks))
        self.assertFalse(preconditioner.blocks[0]._enable_pi_correction)


if __name__ == '__main__':
    unittest.main()
