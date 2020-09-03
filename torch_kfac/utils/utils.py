from typing import Iterable, Tuple
import torch

__all__ = [
    'center',
    'append_homog',
    'compute_cov',
    'power_by_eig_symmetric',
    'inverse_by_cholesky',
    'kronecker_product',
    'normalize_damping',
    'compute_pi_tracenorm',
    'compute_pi_adjusted_damping',
    'inner_product_pairs',
    'scalar_product_pairs'
]

def center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(0, keepdim=True)


def compute_cov(tensor: torch.Tensor, tensor_right: torch.Tensor = None, normalizer=None) -> torch.Tensor:
    """Compute the empirical second moment of the rows of a 2D Tensor.

    This function is meant to be applied to random matrices for which the true row
    mean is zero, so that the true second moment equals the true covariance.

    Args:
    tensor: A 2D Tensor.
    tensor_right: An optional 2D Tensor. If provided, this function computes
        the matrix product tensor^T * tensor_right instead of tensor^T * tensor.
    normalizer: optional scalar for the estimator (by default, the normalizer is
        the number of rows of tensor).

    Returns:
    A square 2D Tensor with as many rows/cols as the number of input columns.
    """
    assert len(tensor.shape) == 2
    if normalizer is None:
        normalizer = tensor.shape[0]

    if tensor_right is None:
        cov = tensor.T @ tensor / normalizer
        # Ensure it is symmetric
        return (cov + cov.T) / 2.0
    else:
        return (tensor.T @ tensor_right) / normalizer


def append_homog(tensor: torch.Tensor, homog_value: float = 1.) -> torch.Tensor:
    """Appends a homogeneous coordinate to the last dimension of a Tensor.

    Args:
    tensor: A Tensor.
    homog_value: Value to append as homogeneous coordinate to the last dimension
        of `tensor`.  (Default: 1.0)

    Returns:
    A Tensor identical to the input but one larger in the last dimension.  The
    new entries are filled with ones.
    """
    shape = list(tensor.shape)
    shape[-1] = 1
    appendage = torch.ones(shape, dtype=tensor.dtype, device=tensor.device) * homog_value
    return torch.cat([tensor, appendage], -1)

def kronecker_product(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    m1, n1 = mat1.shape
    mat1_rsh = mat1.reshape(m1, 1, n1, 1)
    m2, n2 = mat2.shape
    mat2_rsh = mat2.reshape(1, m1, 1, n1)
    return (mat1_rsh * mat2_rsh).reshape(m1*m2, n1*n2)


def normalize_damping(damping: torch.Tensor, num_replications: float, normalize_damping_power: float = 1.) -> torch.Tensor:
    if normalize_damping_power:
        return damping / (num_replications ** normalize_damping_power)
    return damping


def compute_pi_tracenorm(left_cov: torch.Tensor, right_cov: torch.Tensor) -> torch.Tensor:
    left_norm = torch.trace(left_cov) * right_cov.shape[0]
    right_norm = torch.trace(right_cov) * left_cov.shape[0]
    assert torch.all(right_norm > 0), "Pi computation, trace of right cov matrix should be positive!"
    pi = torch.sqrt(left_norm / right_norm)
    return pi


def compute_pi_adjusted_damping(left_cov: torch.Tensor, right_cov: torch.Tensor, damping: torch.Tensor):
    pi = compute_pi_tracenorm(left_cov, right_cov)
    return damping * pi, damping / pi


def power_by_eig_symmetric(tensor: torch.Tensor, damping: torch.Tensor, exp: float) -> torch.Tensor:
    # TODO: If we have multiple exponents and/or dampings we could save the eigenvalues
    eigenvalues, eigenvectors = torch.symeig(tensor, eigenvectors=True)
    return (eigenvectors * (eigenvalues + damping)**exp) @ eigenvectors.T


def inverse_by_cholesky(tensor: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    damped = tensor + torch.eye(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype) * damping
    cholesky = torch.cholesky(damped)
    return torch.cholesky_inverse(cholesky)


def scalar_product_pairs(scalar, list_: Iterable[Tuple[Iterable[torch.Tensor], object]]) -> Iterable[Tuple[Iterable[torch.Tensor], object]]:
    return tuple((tuple(scalar*item for item in items), var) for items, var in list_)

def inner_product(list1: Iterable[torch.Tensor], list2: Iterable[torch.Tensor]) -> torch.Tensor:
    return sum((tensor1 * tensor2).sum() for tensor1, tensor2 in zip(list1, list2))

def inner_product_pairs(list1: Iterable[Tuple[Iterable[torch.Tensor], object]], list2: Iterable[Tuple[Iterable[torch.Tensor], object]]):
    return inner_product(
        tuple(tensor for tensors, _ in list1 for tensor in tensors),
        tuple(tensor for tensors, _ in list2 for tensor in tensors)
    )
