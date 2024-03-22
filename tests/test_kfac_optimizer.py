import unittest

from torch import tensor, float64, double
from torch.nn import Linear
from torch.testing import assert_allclose, assert_close

from torch_kfac import KFAC
from torch_kfac.utils import inverse_by_cholesky


class KfacOptimizerTest(unittest.TestCase):

    def setUp(self):
        self.model = Linear(2, 3, dtype=float64)
        self.input = tensor([[1, 0], [0, 1], [0.5, 0.5], [2, 0.5], [4, 1.1]], dtype=float64)
        self.damping = 1e-1
        self.preconditioner = KFAC(self.model, 0, self.damping)
        self.test_block = self.preconditioner.blocks[0]

    def forward_backward_pass(self):
        with self.preconditioner.track_forward():
            loss = self.model(self.input).norm()
        with self.preconditioner.track_backward():
            loss.backward()

    def calculate_expected_matrices(self):
        """
        Calculates the expected inverse covariance matrices.
        """
        a_damp, s_damp = self.test_block.compute_damping(self.damping, self.test_block.renorm_coeff)
        self.exp_activations_cov_inv = inverse_by_cholesky(self.test_block.activation_covariance, a_damp)
        self.exp_sensitivities_cov_inv = inverse_by_cholesky(self.test_block.sensitivity_covariance, s_damp)

    def test_step_should_calculate_inverses_when_none(self):
        self.forward_backward_pass()
        self.preconditioner.step(update_inverses=False)
        self.calculate_expected_matrices()
        assert_close(self.exp_activations_cov_inv, self.test_block._activations_cov_inv)
        assert_close(self.exp_sensitivities_cov_inv, self.test_block._sensitivities_cov_inv)

    def test_step_should_not_recalculate_inverses(self):
        self.forward_backward_pass()
        self.preconditioner.step()
        self.calculate_expected_matrices()
        self.test_block._activations_cov_inv *= 2
        self.test_block._sensitivities_cov_inv *= 1.5
        self.preconditioner.step(update_inverses=False)
        assert_close(self.exp_activations_cov_inv * 2, self.test_block._activations_cov_inv)
        assert_close(self.exp_sensitivities_cov_inv * 1.5, self.test_block._sensitivities_cov_inv)

    def test_step_should_recalculate_inverses(self):
        self.forward_backward_pass()
        self.preconditioner.step()
        self.calculate_expected_matrices()
        self.test_block._activations_cov_inv *= 2
        self.test_block._sensitivities_cov_inv *= 1.5
        self.preconditioner.step(update_inverses=True)
        assert_close(self.exp_activations_cov_inv, self.test_block._activations_cov_inv)
        assert_close(self.exp_sensitivities_cov_inv, self.test_block._sensitivities_cov_inv)


if __name__ == '__main__':
    unittest.main()
