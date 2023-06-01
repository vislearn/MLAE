from typing import Tuple

import torch.utils.data

from .latent import get_latent_datasets
from .image import get_mnist_datasets, get_cifar10_datasets, get_celeba_datasets
from .toy import make_toy_data

TrainValTest = Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]


def load_dataset(name: str, **kwargs) -> TrainValTest:
    if name in ["miniboone", "gas", "hepmass", "power"]:
        from .tabular import get_tabular_datasets
        # note that the given train/val/test split is ignored and a fixed split is performed
        return get_tabular_datasets(name=name, **kwargs)
    elif name == "mnist":
        return get_mnist_datasets(**kwargs)
    elif name == "cifar10":
        return get_cifar10_datasets(**kwargs)
    elif name == "celeba":
        return get_celeba_datasets(**kwargs)
    elif name in ["latent", "mnist-latent"]:
        return get_latent_datasets(name, **kwargs)
    else:
        return make_toy_data(name, **kwargs)
