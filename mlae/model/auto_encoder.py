from collections import OrderedDict

import torch.nn
from torch import nn
from torch.nn import Module

from .mlae import ModelHParams
from mlae.model.utils import make_dense


class AutoEncoderHParams(ModelHParams):
    layer_spec: list
    skip_connection: bool = False
    detached_latent: bool = False

    def __init__(self, **hparams):
        # Compatibility with old checkpoints
        if "latent_layer_spec" in hparams:
            assert len(hparams["latent_layer_spec"]) == 0
            del hparams["latent_layer_spec"]
        super().__init__(**hparams)


class SkipConnection(Module):
    def __init__(self, inner: Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        return x[..., :out.shape[-1]] + out


class AutoEncoder(nn.Module):
    hparams: AutoEncoderHParams

    def __init__(self, hparams: dict | AutoEncoderHParams):
        if not isinstance(hparams, AutoEncoderHParams):
            hparams = AutoEncoderHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.model = self.build_model()

        if self.hparams.latent_dim != self.hparams.data_dim and self.hparams.skip_connection:
            raise ValueError("Can only have skip connection if data_dim = latent_dim")

    def encode(self, x, c):
        return self.model.encoder(torch.cat([x, c], -1))

    def decode(self, z, c):
        return self.model.decoder(torch.cat([z, c], -1))

    def build_model(self) -> nn.Module:
        data_dim = self.hparams.data_dim
        cond_dim = self.hparams.cond_dim

        # Nonlinear projection
        widths = [
            data_dim,
            *self.hparams.layer_spec,
            self.hparams.latent_dim
        ]
        encoder = make_dense([
            widths[0] + cond_dim,
            *widths[1:]
        ], "silu")
        decoder = make_dense([
            widths[-1] + cond_dim,
            *widths[-2::-1]
        ], "silu")

        modules = OrderedDict(
            encoder=encoder,
            decoder=decoder
        )
        # Apply skip connections
        if self.hparams.skip_connection:
            new_modules = OrderedDict()
            for key, value in modules.items():
                new_modules[key] = SkipConnection(value)
            modules = new_modules
        return torch.nn.Sequential(modules)
