import unittest

from torch import tensor
from torch.nn import Linear

from torch_kfac import KFAC


class KfacOptimizerTest(unittest.TestCase):
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
