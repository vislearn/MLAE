# MIT License
#
# Copyright (c) 2023 Computer Vision and Learning Lab, Heidelberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import namedtuple
from math import sqrt
from typing import Union, Callable

import torch
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

SurrogateOutput = namedtuple("SurrogateOutput", ["z", "x1", "nll", "volume_change"])
Transform = Callable[[torch.Tensor], torch.Tensor]


def sample_v(x: torch.Tensor, hutchinson_samples: int):
    """
    Sample a random vector v of shape (batch_size, x.shape[1], hutchinson_samples)
    with orthonormal columns.

    :param x: Input data. Shape: (batch_size, ...)
    :param hutchinson_samples: Number of Hutchinson samples to draw.
    :return:
    """
    if hutchinson_samples > x.shape[-1]:
        raise ValueError(f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {x.shape[-1]}")
    v = torch.randn(*x.shape, hutchinson_samples, device=x.device, dtype=x.dtype)
    q = torch.linalg.qr(v).Q
    return q * sqrt(x.shape[-1])


def ml_surrogate(x: torch.Tensor, encode: Transform, decode: Transform,
                 hutchinson_samples: int = 1) -> SurrogateOutput:
    """
    Compute the per-sample surrogate for the negative log-likelihood and the volume change estimator.
    The gradient of the surrogate is the gradient of the actual negative log-likelihood.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
    :return: Per-sample loss. Shape: (batch_size,)
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


def mlae_loss(x: torch.Tensor,
              encode: Transform, decode: Transform,
              beta: Union[float | torch.Tensor],
              hutchinson_samples: int = 1) -> torch.Tensor:
    """
    Compute the per-sample MLAE loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 + ||encode(x)||^2 // 2 - \sum_{k=1}^K v_k^T Je stop_grad(Jd) v_k
    $$
    where $E[v_k^T v_k] = 1$, and $Je$ and $Jd$ are the Jacobians of `encode` and `decode`.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    :param beta: Weight of the mean squared error.
    :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
    :return: Per-sample loss. Shape: (batch_size,)
    """
    surrogate = ml_surrogate(x, encode, decode, hutchinson_samples)
    mse = (x - surrogate.x1) ** 2
    return beta * mse + surrogate.nll
