import unittest
from math import sqrt

from torch import float64, tensor, device

from torch_kfac.layers import FisherBlock
from torch_kfac.utils import Lock


class FisherBlockTest(unittest.TestCase):

    # activations covariance matrix
    act_cov = tensor([[1, 2, 3], [2, 5, 6], [3, 6, 9]], dtype=float64) / 10
    # sensitivities covariance matrix
    sen_cov = tensor([[4, 5, 6], [5, 8, 9], [6, 9, 12]], dtype=float64) / 10

    def get_test_block(self) -> FisherBlock:
        block = FisherBlock(3, 3, dtype=float64, device=device("cpu"))
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


if __name__ == '__main__':
    unittest.main()
