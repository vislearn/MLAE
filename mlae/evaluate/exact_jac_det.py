from collections import namedtuple
from functools import wraps

import torch

try:
    from torch.func import vmap, jacrev, jacfwd
except ImportError:
    from functorch import vmap, jacrev, jacfwd


def double_output(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out, out

    return wrapper


ExactOutput = namedtuple("ExactOutput", ["z", "x1", "nll", "log_det"])


def log_det_exact(x, encode, decode, *func_args,
                  grad_type="backward", jacobian_target="encoder",
                  chunk_size=None, allow_gradient=False, **func_kwargs) -> ExactOutput:
    """
    Compute the exact log determinant of the Jacobian of the given encoder or decoder.

    :param x: The input batch.
    :param encode: The encoder function, is called as `encode(x, *func_args, **func_kwargs)`
    :param decode: The decoder function, is called as `decode(z, *func_args, **func_kwargs)`
    :param grad_type: Should the Jacobian be computed using forward or backward mode AD?
    :param jacobian_target: Should the Jacobian of the encoder or decoder be computed?
    :param chunk_size: Set to a positive integer to enable chunking of the computation.
    :param allow_gradient: Should gradients be allowed to flow through the computation?
    :param func_args: Additional arguments to encode/decode
    :param func_kwargs: Additional keyword arguments to encode/decode
    :return:
    """
    if torch.is_grad_enabled() and not allow_gradient:
        raise RuntimeError("Exact log det computation is only recommended in torch.no_grad() mode "
                           "as training may be unstable (see Section 4.2 in the MLAE paper). "
                           "Set log_det_estimator.allow_gradient=True to allow computing gradients.")
    jacfn = jacrev if grad_type == "backward" else jacfwd

    if jacobian_target == "encoder":
        jac, z = vmap(jacfn(double_output(encode), has_aux=True),
                      chunk_size=chunk_size)(x, *func_args, **func_kwargs)
        x1 = decode(z, *func_args, **func_kwargs)
        factor = 1
    elif jacobian_target == "decoder":
        z = encode(x, *func_args, **func_kwargs)
        jac, x1 = vmap(jacfn(double_output(decode), has_aux=True),
                       chunk_size=chunk_size)(z, *func_args, **func_kwargs)
        # Transpose because of inverse direction
        jac = jac.transpose(1, 2)
        factor = -1
    else:
        raise ValueError(f"{jacobian_target=!r}")

    full_dimensional = jac.shape[-1] == jac.shape[-2]
    if full_dimensional:
        jac_log_det = factor * jac.slogdet()[1]
    else:
        jac_transpose_jac = torch.bmm(jac, jac.transpose(1, 2))
        jac_log_det = factor * jac_transpose_jac.slogdet()[1] / 2

    nll = torch.sum((z ** 2), -1) / 2 - jac_log_det

    return ExactOutput(
        z, x1, nll, jac_log_det,
    )
