# Maximum Likelihood Training of Autoencoders

![figs/mlae-overview.png](figs/mlae-overview.png)

This is the official `PyTorch` implementation of [our preprint](http://arxiv.org/abs/2306.01843):

```bibtex
@article{sorrenson2023maximum,
    title = {Maximum Likelihood Training of Autoencoders},
    author = {Sorrenson, Peter and Draxler, Felix and Rousselot, Armand and Hummerich, Sander and Zimmermann, Lea and Köthe, Ullrich},
    journal = {arXiv preprint arXiv:2306.01843},
    year = {2023}
}
```

## Installation

### Install via pip

The following will install our package along with all of its dependencies:

```bash
git clone https://github.com/vislearn/MLAE.git
cd MLAE
pip install -r requirements.txt
pip install .
```

Then you can import the package via

```python
import mlae
```

### Copy `mlae/loss.py` into your project

If you do not want to add our `mlae` package as a dependency,
but still want to use the MLAE loss function,
you can copy the `mlae/loss.py` file into your own project.
It does not have any dependencies on the rest of the repo.


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

Our training framework is built on https://github.com/LarsKue/lightning-trainable. 
You can train all our models via the `lightning_trainable.launcher.fit` module.
For example, to train our MNIST model:
```bash
python -m lightning_trainable.launcher.fit configs/mnist.yaml --name '{data_set[name]},{models[0][latent_dim]}'
```

This will create a new directory `lightning_logs/mnist,16/`. You can monitor the run via `tensorboard`:
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

If you want to overwrite the default parameters, you can add `key=value`-pairs after the config file:
```bash
python -m lightning_trainable.launcher.fit configs/mnist.yaml batch_size=128 loss_weights.noisy_reconstruction=20 --name '{data_set[name]},{models[0][latent_dim]}'
```


## Reproduce our experiments

To reproduce an experiment in the paper, use the provided config files.
For some experiments, we vary parameters to demonstrate their effect.
You can set them via `key=value` pairs:

```bash
python -m lightning_trainable.launcher.fit [config file(s)] [key=value pairs] --name '{data_set[name]},{models[0][latent_dim]}'
```

| Experiment        | Configuration specification                                                             |
|-------------------|-----------------------------------------------------------------------------------------|
| Toy data          | `configs/toy.yaml loss_weights.noisy_reconstruction=… noise=…`                          |
| Tabular           | `configs/tabular.yaml configs/tabular-….yaml`                                           |
| Conditional MNIST | `configs/mnist.yaml configs/mnist-conditional.yaml loss_weights.noisy_reconstruction=…` |
| MNIST             | `configs/mnist.yaml`                                                                    |
| CelebA            | `configs/celeba.yaml`                                                                   |
