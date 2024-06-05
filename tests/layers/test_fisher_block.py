import unittest
from math import sqrt
from typing import Iterable
from unittest.mock import MagicMock

import torch
from torch import float64, tensor, device, allclose, cat, zeros, rand
from torch.testing import assert_allclose, assert_close

from torch_kfac.layers import FisherBlock
from torch_kfac.utils import Lock


class MockFisherBlock(FisherBlock):
    def mat_to_grads(self, mat_grads: torch.Tensor) -> torch.Tensor:
        return mat_grads

    def grads_to_mat(self, grads: Iterable[torch.Tensor]) -> torch.Tensor:
        return cat(tuple(grads))


class FisherBlockTest(unittest.TestCase):
    # activations covariance matrix
    act_cov = tensor([[1, 2, 3], [2, 5, 6], [3, 6, 9]], dtype=float64) / 10
    # sensitivities covariance matrix
    sen_cov = tensor([[4, 5, 6], [5, 8, 9], [6, 9, 12]], dtype=float64) / 10

    def get_test_block(self) -> FisherBlock:
        block = MockFisherBlock(3, 3, dtype=float64, device=device("cpu"))
        block._activations_cov.add_to_average(self.act_cov, decay=0)
        block._sensitivities_cov.add_to_average(self.sen_cov, decay=0)
        return block

    def test_calculate_damping_with_pi_correction(self):
        """
        Tests calculation of damping if Tikhonov pi correction is enabled.
        """
        block = self.get_test_block()
        block.setup(Lock(), Lock(), True)
        damping = 1e-1
        a_damp, s_damp = block.compute_damping(tensor(damping), block.renorm_coeff)

        denominator = (4 + 8 + 12) / 10 * 3
        numerator = (1 + 5 + 9) / 10 * 3
        pi = sqrt(numerator / denominator)
        expected_a_damp = damping ** 0.5 * pi
        expected_s_damp = damping ** 0.5 / pi
        self.assertAlmostEqual(expected_a_damp, a_damp.item())
        self.assertAlmostEqual(expected_s_damp, s_damp.item())

    def test_calculate_damping_no_pi_correction(self):
        """
        Tests calculation of damping if Tikhonov pi correction is disabled.
        """
        block = self.get_test_block()
        block.setup(Lock(), Lock(), False)
        damping = 1e-1
        a_damp, s_damp = block.compute_damping(tensor(damping), block.renorm_coeff)
        expected_damp = damping ** 0.5
        self.assertAlmostEqual(expected_damp, a_damp.item())
        self.assertAlmostEqual(expected_damp, s_damp.item())

    def test_multiply_preconditioner_check_result(self):
        """Tests if a random gradient is preconditioned correctly."""
        block = self.get_test_block()

        expected_sens_cov_inv = tensor([[1.8624, -0.4362, -0.4530],
                                        [-0.4362, 1.5436, -0.7047],
                                        [-0.4530, -0.7047, 1.1913]], dtype=float64)

        expected_act_cov_inv = tensor([[3.7395, -0.3721, -0.7814],
                                       [-0.3721, 2.3256, -1.1163],
                                       [-0.7814, -1.1163, 1.6558]], dtype=float64)
        test_grads = [rand((3, 3), dtype=float64)]
        damping = tensor(1e-1)
        actual_preconditioned_grad = block.multiply_preconditioner(test_grads, damping)
        expected_preconditioned_grad = expected_sens_cov_inv @ test_grads[0] @ expected_act_cov_inv / block.renorm_coeff
        assert_close(actual_preconditioned_grad, expected_preconditioned_grad, rtol=1e-4, atol=1e-4)

    def test_apply_preconditioner_should_not_recalculate_inverses(self):
        """
        Tests that the inverse covariance matrices are not recalculated
        when the multiply_preconditioner method is called twice.
        """
        block = self.get_test_block()
        test_grads = [rand((3, 3), dtype=float64)]
        damping = tensor(1e-1)
        block.multiply_preconditioner(test_grads, damping)

        block.update_cov_inv = MagicMock()
        block.multiply_preconditioner(test_grads, damping)
        self.assertFalse(block.update_cov_inv.called)


if __name__ == '__main__':
    unittest.main()
