from mlae.loss import mlae_loss
import numpy as np
import torch
import torch.nn as nn

D = 2
d = 1
stdevs = torch.arange(1, D+1)
encoder = nn.Linear(D, d)
decoder = nn.Linear(d, D)
beta = 10
batch_size = 1000
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-3)
for i in range(10000):
    x = torch.randn(batch_size, D) * stdevs[None]
    loss = mlae_loss(x, encoder, decoder, beta).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test_convergence():
    assert np.allclose(abs(encoder.weight.detach().squeeze().numpy()), np.array([0, 0.5]), atol=1e-2)
    assert np.allclose(abs(decoder.weight.detach().squeeze().numpy()), np.array([0, 2]), atol=1e-2)