from mlae.loss import mlae_loss
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

D = 10
d = 5
batch_size = 2
x = torch.randn(batch_size, D)
encoder = nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, d))
decoder = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, D))
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
beta = 1.0

def surrogate_term_per_element(xi, encoder, decoder):
    Je = jacobian(encoder, xi[None], create_graph=True).squeeze()
    Jd = jacobian(decoder, encoder(xi[None]), create_graph=False).squeeze().detach()
    return torch.trace(Je @ Jd)

def exact_loss_function(x, encoder, decoder, beta):
    z = encoder(x)
    x_hat = decoder(z)
    s = torch.cat([surrogate_term_per_element(xi, encoder, decoder)[None] for xi in x])
    return 0.5 * torch.sum(z**2, -1) - s + beta * torch.sum((x - x_hat)**2, -1)

def test_gradient_to_enc_and_dec():
    optimizer.zero_grad()
    loss = mlae_loss(x, encoder, decoder, beta, hutchinson_samples=1).mean()
    loss.backward()
    assert all([p.grad is not None for p in encoder.parameters()]) 
    assert all([p.grad is not None for p in decoder.parameters()])

def test_surrogate_gradient_accuracy():
    # Compare the surrogate gradient to the exact gradient when huitchinson_samples=d
    optimizer.zero_grad()
    loss = mlae_loss(x, encoder, decoder, beta, hutchinson_samples=d).mean()
    loss.backward()
    grads_est = torch.cat([p.grad.flatten() for p in encoder.parameters()] + [p.grad.flatten() for p in decoder.parameters()]).numpy()
    optimizer.zero_grad()
    loss = exact_loss_function(x, encoder, decoder, beta).mean()
    loss.backward()
    grads_exact = torch.cat([p.grad.flatten() for p in encoder.parameters()] + [p.grad.flatten() for p in decoder.parameters()]).numpy()
    assert np.allclose(grads_est, grads_exact, atol=1e-6)
