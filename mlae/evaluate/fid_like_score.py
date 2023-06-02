import os
import torch
import numpy as np
from mlae.model import MaximumLikelihoodAutoencoder
from mlae.model.mlae import wasserstein2_distance_gaussian_approximation

@torch.no_grad()
def fid_like_score_test_data(checkpoint):
    # cd to the MLAE directory to load the model then back to the original directory
    current_dir = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir.split('mlae')[0])
    model = MaximumLikelihoodAutoencoder.load_from_checkpoint(checkpoint)
    os.chdir(current_dir)
    x_test = model.test_data.tensors[0].to(model.device)
    # sample from the prior and pass through the decoder
    z_sample = torch.randn(x_test.shape[0], model.latent_dim, device=model.device)
    c1 = model.apply_conditions([z_sample]).condition
    x_sample = model.decode(z_sample, c1).cpu()
    return wasserstein2_distance_gaussian_approximation(x_sample, x_test.cpu())