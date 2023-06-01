from collections import namedtuple
from math import sqrt

import torch
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

SurrogateOutput = namedtuple("SurrogateOutput", ["z", "x1", "nll", "volume_change"])


def sample_v(x, hutchinson_samples):
    if hutchinson_samples > x.shape[-1]:
        raise ValueError(f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {x.shape[-1]}")
    v = torch.randn(*x.shape, hutchinson_samples, device=x.device, dtype=x.dtype)
    q = torch.linalg.qr(v).Q
    return q * sqrt(x.shape[-1])


def ml_surrogate(x, encode, decode, hutchinson_samples) -> SurrogateOutput:
    """
    Compute the per-sample surrogate for the negative log-likelihood and the volume change estimator.
    The gradient of the surrogate is the gradient of the actual negative log-likelihood.

    :param x:
    :param encode:
    :param decode:
    :param hutchinson_samples:
    :return:
    """
    x.requires_grad_()
    z = encode(x)

    s = 0
    vs = sample_v(z, hutchinson_samples)
    for k in range(hutchinson_samples):
        v = vs[..., k]

        # $ Jd v $ via forward-mode AD
        with dual_level():
            dual_z = make_dual(z, v)
            dual_x1 = decode(dual_z)
            x1, v1 = unpack_dual(dual_x1)

        # $ v^T Je $ via backward-mode AD
        v2, = grad(z, x, v, create_graph=True)

        # $ v^T Je stop_grad(Jd) v $
        s += torch.sum(v2 * v1.detach(), -1) / hutchinson_samples

    # Per-sample negative log-likelihood
    ml = (z ** 2) / 2 - s

    return SurrogateOutput(z, x1, ml, s)


def mlae_loss(x, encode, decode, beta, hutchinson_samples) -> torch.Tensor:
    """
    Compute the per-sample MLAE loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 + ||encode(x)||^2 // 2 - \sum_{k=1}^K v_k^T Je stop_grad(Jd) v_k
    $$
    where $E[v_k^T v_k] = 1$, and $Je$ and $Jd$ are the Jacobians of `encode` and `decode`.

    :param x:
    :param encode:
    :param decode:
    :param beta:
    :param hutchinson_samples:
    :return: Per-sample loss. Shape: (batch_size,)
    """
    surrogate = ml_surrogate(x, encode, decode, hutchinson_samples)
    mse = (x - surrogate.x1) ** 2
    return beta * mse + surrogate.nll
