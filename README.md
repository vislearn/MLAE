# Maximum Likelihood Training of Autoencoders

This is the official `PyTorch` implementation of our preprint:

```bibtex
@article{sorrenson2023maximum,
    title = {Maximum Likelihood Training of Autoencoders},
    author = {Sorrenson, Peter and Draxler, Felix and Rousselot, Armand and Hummerich, Sander and Zimmermann, Lea and Köthe, Ullrich},
    journal = {arXiv preprint arXiv:2306.XXXXX},
    year = {2023}
}
```

## Basic usage

### Train your architecture 

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

### Build models based on our framework

If you want to build your models based on our framework, you need to install
`lightning_trainable` from their [GitHub repo](https://github.com/LarsKue/lightning-trainable)
via `pip`.
This will automatically install the required dependencies, including PyTorch-Lightning.

This will create a new directory `lightning_logs/mnist,16/`. You can trace the run via `tensorboard`:
```bash
tensorboard --logdir lightning_logs
```

When training has finished, you can import the model via
```python
import mlae

model = mlae.model.MaximumLikelihoodAutoencoder.load_from_checkpoint(
    'lightning_logs/mnist,16/version_0/checkpoints/last.ckpt'
)
```

You can train all our models via the `lightning_trainable.launcher.fit` module.
For example, to train our MNIST model:
```bash
python -m lightning_trainable.launcher.fit configs/mnist.yaml --name '{data_set[kind]},{models[0][latent_dim]}'
```

If you want to overwrite the default parameters, you can add them after the config file:
```bash
python -m lightning_trainable.launcher.fit configs/mnist.yaml batch_size=128 loss_weights.noisy_reconstruction=20 --name '{data_set[kind]},{models[0][latent_dim]}'
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

## Reproduce our experiments

To reproduce an experiment in the paper, you can use our provided config files.
For some experiments, we vary parameters to demonstrate their effect. You can set them via `key=value` pairs:

```bash
python -m lightning_trainable.launcher.fit [config file(s)] [key=value pairs] --name '{data_set[kind]},{models[0][latent_dim]}'
```

| Experiment        | Configuration specification                                                             |
|-------------------|-----------------------------------------------------------------------------------------|
| Toy data          | `configs/toy.yaml loss_weights.noisy_reconstruction=… noise=…`                          |
| Tabular           | `configs/tabular.yaml configs/tabular-….yaml`                                           |
| Conditional MNIST | `configs/mnist.yaml configs/mnist-conditional.yaml loss_weights.noisy_reconstruction=…` |
| MNIST             | `configs/mnist.yaml`                                                                    |
| CelebA            | `configs/celeba.yaml`                                                                   |