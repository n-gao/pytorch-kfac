import unittest

import torch
from numpy import sqrt
from torch import tensor, float64, allclose, device

from torch_kfac.layers import FisherBlock


class FisherBlockTest(unittest.TestCase):
    # activations covariance matrix
    act_cov = tensor([[1, 2, 3], [2, 5, 6], [3, 6, 9]], dtype=torch.float64) / 10
    # sensitivities covariance matrix
    sen_cov = tensor([[4, 5, 6], [5, 8, 9], [6, 9, 12]], dtype=torch.float64) / 10

    def get_test_block(self) -> FisherBlock:
        block = FisherBlock(3, 3, dtype=float64, device=device("cpu"))
        block._activations_cov.add_to_average(self.act_cov, decay=0)
        block._sensitivities_cov.add_to_average(self.sen_cov, decay=0)
        return block

    def test_calculate_damping(self):
        block = self.get_test_block()
        damping = 1e-1
        a_damp, s_damp = block.compute_damping(tensor(damping), block.renorm_coeff)

        denominator = (4 + 8 + 12) / 10 * 3
        numerator = (1 + 5 + 9) / 10 * 3
        pi = sqrt(numerator / denominator)
        expected_a_damp = damping ** 0.5 * pi
        expected_s_damp = damping ** 0.5 / pi
        self.assertAlmostEqual(expected_a_damp, a_damp.item())
        self.assertAlmostEqual(expected_s_damp, s_damp.item())

    def test_inverse_calculation_check_sensitivities(self):
        block = self.get_test_block()
        damping = 1e-1
        block.update_cov_inv(tensor(damping))
        expected_sens_cov_inv = tensor([[1.8624, -0.4362, -0.4530],
                                        [-0.4362, 1.5436, -0.7047],
                                        [-0.4530, -0.7047, 1.1913]], dtype=float64)

        self.assertTrue(allclose(expected_sens_cov_inv, block._sensitivities_cov_inv, rtol=1e-4))

    def test_inverse_calculation_check_activations(self):
        block = self.get_test_block()
        damping = 1e-1
        block.update_cov_inv(tensor(damping))

        expected_act_cov_inv = tensor([[3.7395, -0.3721, -0.7814],
                                       [-0.3721, 2.3256, -1.1163],
                                       [-0.7814, -1.1163, 1.6558]], dtype=float64)

        self.assertTrue(allclose(expected_act_cov_inv, block._activations_cov_inv, rtol=1e-4))


if __name__ == '__main__':
    unittest.main()
