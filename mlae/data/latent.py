from importlib import import_module

import torch
from torch.utils.data import TensorDataset


class LatentDataset(TensorDataset):
    def __init__(self, model, base_dataset: TensorDataset):
        super().__init__(torch.cat([
            model.encode(batch)
            for batch in base_dataset.tensors[0].split(model.hparams.batch_size)
        ]))
        self.model = model


@torch.no_grad()
def get_latent_datasets(checkpoint: str, model: str = "mlae.model.ConvAutoEncoder"):
    module_name, model_name = model.rsplit(".", 1)
    module = import_module(module_name)

    model_class = getattr(module, model_name)
    model = model_class.load_from_checkpoint(checkpoint)

    return tuple(
        LatentDataset(model, data_set)
        for data_set in [model.train_data, model.val_data, model.test_data]
    )
