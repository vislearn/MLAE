# Maximum Likelihood Training of Autoencoders

```bibtex
@article{sorrenson2023maximum,
    title = {Maximum Likelihood Training of Autoencoders},
    author = {Sorrenson, Peter and Draxler, Felix and Rousselot, Armand and Hummerich, Sander and Zimmermann, Lea and Köthe, Ullrich},
    journal = {arXiv preprint arXiv:2306.XXXXX},
    year = {2023}
}
```

## Basic usage

```python

import torch
import mlae.loss as loss


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(...)
        self.decoder = torch.nn.Sequential(...)


model = AutoEncoder()
optim = ...
data_loader = ...
n_epochs = ...
beta = ...

for epoch in range(n_epochs):
    for batch in data_loader:
        optim.zero_grad()
        loss = loss.mlae_loss(batch, model.encoder, model.decoder, beta)
        loss.backward()
        optim.step()
```

## Installation

### Install via pip

```bash
git clone https://github.com/vislearn/mlae.git
cd mlae
pip install .
```

Then you can import the package via

```python
import mlae
```

### Copy `mlae/loss.py` into your project

If you do not want to add our `mlae` package as a dependency,
you can also copy the `mlae/loss.py` file into your own project.
It only depends on `torch`.

## Reproducing the experiments

This section is work in progress. We will add the full code
to reproduce the experiments in the paper soon.
