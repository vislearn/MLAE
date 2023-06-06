from math import sqrt

import torch
from torch.autograd import grad
import torch.autograd.forward_ad as fwAD
from torch.autograd.forward_ad import make_dual, unpack_dual, dual_level


def sample_v(x, hutchinson_samples, orthogonalize=True):
    if hutchinson_samples > x.shape[-1]:
        raise ValueError(
            f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {x.shape[-1]}"
        )
    v = torch.randn(*x.shape, hutchinson_samples, device=x.device, dtype=x.dtype)
    q = (
        torch.linalg.qr(v).Q
        if orthogonalize
        else v / torch.linalg.norm(v, axis=1).unsqueeze(1)
    )
    return q * sqrt(x.shape[-1])


def log_det_surrogate_data_encoder_mixed(
    x, c, encode, decode, hutchinson_samples, orthogonalize=True
):
    x.requires_grad_()
    z = encode(x, c)
    x1 = decode(z, c)

    s = 0
    vs = sample_v(x, hutchinson_samples, orthogonalize=orthogonalize)
    for k in range(hutchinson_samples):
        v = vs[..., k]
        v1 = grad(x1, z, v, retain_graph=True)[0]
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(x1, v)
            dual_output = encode(dual_input, c)
            z1, v2 = fwAD.unpack_dual(dual_output)
        s += torch.sum(v2 * v1.detach(), -1) / hutchinson_samples
    assert len(s.shape) == 1
    return z, x1, s, s.detach()


def log_det_surrogate_latent_encoder_mixed(
    x, c, encode, decode, hutchinson_samples, orthogonalize=True
):
    x.requires_grad_()
    z = encode(x, c)

    s = 0
    s_all_grad = 0
    vs = sample_v(z, hutchinson_samples, orthogonalize=orthogonalize)
    for k in range(hutchinson_samples):
        v = vs[..., k]
        # $ SG(Jd) v $
        with dual_level():
            dual_z = make_dual(z, v)
            dual_x1 = decode(dual_z, c)
            x1, v1 = unpack_dual(dual_x1)

        (v2,) = grad(z, x, v, create_graph=True)
        s += torch.sum(v2 * v1.detach(), -1) / hutchinson_samples
        s_all_grad += torch.sum(v2 * v1, -1) / hutchinson_samples

    return z, x1, s, s_all_grad
