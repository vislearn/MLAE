from collections import defaultdict, namedtuple
from copy import deepcopy
from importlib import import_module
from math import log10

import numpy as np
import torch
from FrEIA.distributions import StandardNormalDistribution
from lightning_trainable import TrainableHParams, Trainable
from lightning_trainable.hparams import HParams
from lightning_trainable.trainable import SkipBatch
from matplotlib import pyplot as plt
from torch.nn import Sequential

from mlae.data import load_dataset
from mlae.data.multivariate_student_t import MultivariateStudentT
from mlae.evaluate import plot_manifold, plot_latent_codes, make_img_samples_grid, \
    make_img_reconstruction_grid, SkipModel
from mlae.evaluate.exact_jac_det import log_det_exact
from mlae.loss import nll_surrogate
from mlae.evaluate.mmd import maximum_mean_discrepancy


class ModelHParams(HParams):
    data_dim: int
    cond_dim: int
    latent_dim: int


LogProbResult = namedtuple("LogProbResult", ["z", "x1", "log_prob"])
ConditionedBatch = namedtuple("ConditionedBatch", ["x0", "x_noisy", "loss_weights", "condition"])


class MLAEHParams(TrainableHParams):
    models: list

    log_det_estimator: dict = dict(
        name="surrogate",
        hutchinson_samples=1
    )
    skip_val_nll: bool | int = False

    noise: float | list = 0.0
    loss_weights: dict
    warm_up_epochs: int | list = 0
    latent_distribution: dict = dict(
        name="normal"
    )

    exact_chunk_size: None | int = None

    data_set: dict

    def __init__(self, **hparams):
        if "models" not in hparams:
            if "latent_inn_spec" in hparams:
                model_class = "mlae.model.LatentFlow"
                copy_keys = ["layer_spec", "latent_inn_spec", "detached_inn"]
            elif "layers_spec" in hparams:
                model_class = "mlae.model.ResNet"
                copy_keys = ["layers_spec", "latent_spec", "activation"]
            elif "zero_init" in hparams:
                model_class = "mlae.model.SurFlow"
                copy_keys = ["inn_spec", "zero_init"]
            elif "latent_layer_spec" in hparams:
                model_class = "mlae.model.AutoEncoder"
                copy_keys = [
                    "layer_spec",
                    "latent_layer_spec",
                    "skip_connection",
                    "detached_latent"
                ]
            elif "ch_factor" in hparams:
                model_class = "mlae.model.ConvAutoEncoder"
                copy_keys = [
                    "skip_connection",
                    "ch_factor",
                    "encoder_spec",
                    "encoder_fc_spec",
                    "decoder_fc_spec",
                    "decoder_spec",
                    "batch_norm",
                ]
            else:
                model_class = copy_keys = None
            if model_class is not None:
                hparams["models"] = [{
                    key: hparams.pop(key)
                    for key in copy_keys + ["latent_dim"]
                    if key in hparams
                }]
                hparams["models"][0]["name"] = model_class

        if "log_det_estimator" in hparams and isinstance(hparams["log_det_estimator"], str):
            hparams["log_det_estimator"] = dict(
                log_det_estimator=hparams["log_det_estimator"],
                trace_space=hparams.pop("trace_space", "latent"),
                grad_to_enc_or_dec=hparams.pop("grad_to_enc_or_dec"),
                grad_type=hparams.pop("grad_type"),
                hutchinson_samples=hparams.pop("hutchinson_samples", 1),
            )
            if "detach_non_grad" in hparams:
                hparams["log_det_estimator"]["detach_non_grad"] = hparams.pop("detach_non_grad")
        if "detached_decoder" in hparams and not hparams["detached_decoder"]:
            del hparams["detached_decoder"]

        super().__init__(**hparams)


class MaximumLikelihoodAutoencoder(Trainable):
    hparams: MLAEHParams

    def __init__(self, hparams: MLAEHParams | dict):
        if not isinstance(hparams, MLAEHParams):
            hparams = MLAEHParams(**hparams)

        train_data, val_data, test_data = load_dataset(**hparams.data_set)

        super().__init__(hparams, train_data=train_data, val_data=val_data, test_data=test_data)

        data_sample = train_data[0]
        if len(data_sample[0].shape) != 1:
            raise NotImplementedError("Data must be shaped as a vector.")
        self._data_dim = data_sample[0].shape[0]
        if len(data_sample) == 1:
            self._data_cond_dim = 0
        else:
            if len(data_sample[1].shape) != 1:
                raise NotImplementedError("More than one condition dimension is not supported.")
            self._data_cond_dim = data_sample[1].shape[0]
        cond_dim = self.cond_dim
        self.models = build_model(self.hparams.models, data_sample[0].shape[0], cond_dim)
        self.latents = {}

    def get_latent(self, device):
        if device not in self.latents:
            if self.hparams.latent_distribution["name"] == "normal":
                self.latents[device] = StandardNormalDistribution(self.latent_dim, device=device)
            elif self.hparams.latent_distribution["name"] == "student_t":
                df = self.hparams.latent_distribution["df"] * torch.ones(1, device=device)
                self.latents[device] = MultivariateStudentT(df, self.latent_dim)
            else:
                raise ValueError(f"Unknown latent distribution: {self.hparams.latent_distribution['name']}")
        return self.latents[device]

    @property
    def latent_dim(self):
        return self.models[-1].hparams.latent_dim

    def is_conditional(self):
        return self._data_cond_dim > 1

    @property
    def cond_dim(self):
        soft_flow_cond_dim = 1 if isinstance(self.hparams.noise, list) else 0
        hp_aware_cond_dim = sum(
            1 if isinstance(weight, list) else 0
            for weight in self.hparams.loss_weights.values()
        )

        return self._data_cond_dim + soft_flow_cond_dim + hp_aware_cond_dim

    @property
    def data_dim(self):
        return self._data_dim

    def encode(self, x, c):
        for model in self.models:
            x = model.encode(x, c)
        return x

    def decode(self, z, c):
        for model in self.models[::-1]:
            z = model.decode(z, c)
        return z

    def forward(self, x, c):
        return self.decode(self.encode(x, c), c)

    def log_prob(self, x, c, estimate=False, **kwargs) -> LogProbResult:
        # Then compute JtJ
        config = deepcopy(self.hparams.log_det_estimator)
        estimator_name = config.pop("name")

        if estimate:
            kwargs.update(config)

        if estimator_name == "exact" or not estimate:
            out = log_det_exact(
                x, self.encode, self.decode, c,
                chunk_size=self.hparams.exact_chunk_size,
                **kwargs,
            )
            volume_change = out.log_det
        elif estimator_name == "surrogate":
            out = nll_surrogate(
                x,
                lambda _x: self.encode(_x, c),
                lambda z: self.decode(z, c),
                **kwargs
            )
            volume_change = out.surrogate
        else:
            raise ValueError(f"Cannot understand log_det_estimator.name={estimator_name!r}")

        # Compute log-likelihood
        loss_gauss = self.get_latent(x.device).log_prob(out.z)
        return LogProbResult(
            out.z, out.x1, loss_gauss + volume_change
        )

    def compute_metrics(self, batch, batch_idx) -> dict:
        """
        Computes the metrics for the given batch.

        Rationale:
        - In training, we only compute the terms that are actually used in the loss function.
        - During validation, all possible terms and metrics are computed.

        :param batch:
        :param batch_idx:
        :return:
        """
        x0, x, loss_weights, c = self.apply_conditions(batch)
        loss_values = {}
        metrics = {}

        def check_keys(*keys):
            return any(
                (loss_key in loss_weights)
                and
                (
                    torch.any(loss_weights[loss_key] > 0)
                    if torch.is_tensor(loss_weights[loss_key]) else
                    loss_weights[loss_key] > 0
                )
                for loss_key in keys
            )

        # Empty until computed
        x1 = z = None

        # Negative log-likelihood
        if not self.training:
            if self.hparams.skip_val_nll is False or (
                    isinstance(self.hparams.skip_val_nll, int)
                    and batch_idx < self.hparams.skip_val_nll
            ):
                z, x1, log_prob = self.log_prob(x=x, c=c)
                loss_values["nll"] = -log_prob
            else:
                loss_weights["nll"] = 0
        elif check_keys("nll"):
            warm_up = self.hparams.warm_up_epochs
            if isinstance(warm_up, int):
                warm_up = warm_up, warm_up + 1
            nll_start, warm_up_end = warm_up
            if nll_start == 0:
                nll_warmup = 1
            else:
                nll_warmup = soft_heaviside(
                    self.current_epoch + batch_idx / len(
                        self.trainer.train_dataloader
                        if self.training else
                        self.trainer.val_dataloaders
                    ),
                    nll_start, warm_up_end
                )
            loss_weights["nll"] *= nll_warmup
            if check_keys("nll"):
                z, x1, log_prob = self.log_prob(x=x, c=c, estimate=True)
                loss_values["nll"] = -log_prob

        # In case they were skipped above
        if z is None:
            z = self.encode(x, c)
        if x1 is None:
            x1 = self.decode(z, c)

        if not self.training or check_keys("mmd"):
            latent_samples = self.get_latent(z.device).sample(z.shape[:1])
            loss_values["mmd"] = maximum_mean_discrepancy(latent_samples, z)

        # Wasserstein distance of marginal to Gaussian
        with torch.no_grad():
            z_marginal = z.reshape(-1)
            z_gauss = torch.randn_like(z_marginal)

            z_marginal_sorted = z_marginal.sort().values
            z_gauss_sorted = z_gauss.sort().values

            metrics["z 1D-Wasserstein-1"] = (z_marginal_sorted - z_gauss_sorted).abs().mean()
            metrics["z std"] = torch.std(z_marginal)

        # Reconstruction
        if not self.training or check_keys("reconstruction", "noisy_reconstruction"):
            loss_values["reconstruction"] = reconstruction_loss(x0, x1)
            loss_values["noisy_reconstruction"] = reconstruction_loss(x, x1)

        # Cyclic consistency of latent code
        if not self.training or check_keys("z_reconstruction"):
            # Not reusing x1 from above, as it does not detach z
            z1 = self.encode(x1, c)
            loss_values["z_reconstruction"] = reconstruction_loss(z, z1)

        # Cyclic consistency of latent code -- gradient only to encoder
        if not self.training or check_keys("z_reconstruction_encoder"):
            # Not reusing x1 from above, as it does not detach z
            x1_detached = x1.detach()
            z1 = self.encode(x1_detached, c)
            loss_values["z_reconstruction_encoder"] = reconstruction_loss(z, z1)

        # Cyclic consistency of latent code sampled from Gauss
        if not self.training or check_keys("gauss_z_reconstruction"):
            z_gauss = torch.randn(x.shape[0], self.latent_dim, device=x.device, dtype=x.dtype)
            # We re-use the data noise distribution here, reconstruction should
            # work for all data noises
            x_sample = self.decode(z_gauss, c)
            z_gauss1 = self.encode(x_sample, c)
            loss_values["gauss_z_reconstruction"] = reconstruction_loss(z_gauss, z_gauss1)

        # Reconstruction of Gauss with double std -- for invertibility
        if not self.training or check_keys("gauss_reconstruction"):
            # As we only care about the reconstruction, can ignore noise scale
            x_gauss = 2 * torch.randn_like(x)
            z_gauss = self.encode(x_gauss, c)
            x_gauss1 = self.decode(z_gauss, c)
            loss_values["gauss_reconstruction"] = reconstruction_loss(x_gauss, x_gauss1)

        # Reconstruction of Gauss with double std -- for invertibility
        if not self.training or check_keys("shuffled_reconstruction"):
            # Make noise scale independent of applied noise, reconstruction should still be fine
            x_shuffled = x[torch.randperm(x.shape[0])]
            z_shuffled = self.encode(x_shuffled, c)
            x_shuffled1 = self.decode(z_shuffled, c)
            loss_values["shuffled_reconstruction"] = reconstruction_loss(x_shuffled, x_shuffled1)

        # Compute loss as weighted loss
        metrics["loss"] = sum(
            (weight * loss_values[key]).mean(-1)
            for key, weight in loss_weights.items()
            if check_keys(key)
        )

        # Metrics are averaged, non-weighted loss_values
        for key, weight in loss_values.items():
            # One value per key
            assert loss_values[key].shape == (x.shape[0],)
            metrics[key] = loss_values[key].mean(-1)

        # Store loss weights
        if self.training:
            for key, weight in loss_weights.items():
                if not torch.is_tensor(weight):
                    weight = torch.tensor(weight)
                self.log(f"weights/{key}", weight.float().mean())

        # $Je Jd = I$ assumption on manifold
        if not self.training and batch_idx == 0 and False:
            # This should be a vmapped jacfwd, but it is not implemented for vmap for our model
            Jd = torch.stack([
                torch.func.jacfwd(self.decode)(zi, ci)
                for zi, ci in zip(z, c)
            ])

            x1 = self.decode(z, c)
            Je = torch.func.vmap(torch.func.jacrev(self.encode))(x1, c)
            JJt = torch.bmm(Je, Jd)

            orthogonality = reconstruction_loss(
                torch.eye(JJt.shape[-1], device=x.device).reshape(-1).unsqueeze(0),
                JJt.reshape(x.shape[0], -1)
            )
            metrics["orthogonality"] = orthogonality.mean()
            metrics["orthogonality-std"] = orthogonality.std()

        # Check finite loss
        if not torch.isfinite(metrics["loss"]) and self.training:
            self.trainer.save_checkpoint("erroneous.ckpt")
            print(f"Encountered nan loss from: {metrics}!")
            raise SkipBatch

        if not self.training and batch_idx == 0:
            # compute FID-like score for tabular data
            if self.hparams.data_set["name"] in ["miniboone", "gas", "hepmass", "power"]:
                x_val = self.val_data.tensors[0].to(self.device)
                z_sample = torch.randn(x_val.shape[0], self.latent_dim, device=self.device)
                c1 = self.apply_conditions([z_sample]).condition
                sample = self.decode(z_sample, c1)
                metrics["fid_like_score"] = wasserstein2_distance_gaussian_approximation(sample, x_val)

        return metrics

    def on_train_epoch_end(self) -> None:
        if self.data_dim > 2 and self.hparams.data_set["name"] in ["mnist", "cifar10", "celeba"]:
            with torch.no_grad():
                n_row = 1 if self.is_conditional() else 10
                for temperature in [.5, .8, 1.0, 1.2, 1.5]:
                    if temperature == "z std":
                        temperature_val = self.trainer.callback_metrics["z std"]
                    else:
                        temperature_val = temperature

                    # Log grid of samples
                    sample_img_grid = make_img_samples_grid(self, temperatures=[temperature_val], n_rows=n_row,
                                                            random_state=413978)
                    self.logger.experiment.add_image(f"sample-{temperature}", sample_img_grid, self.global_step)

                # Log grid of reconstructions
                reconstruction_img = make_img_reconstruction_grid(self)
                self.logger.experiment.add_image("reconstruction", reconstruction_img, self.global_step)

        if self.current_epoch % 100 == 99:
            data_dim = self.data_dim
            plot_fns = [plot_latent_codes]
            if data_dim == 2:
                plot_fns.append(plot_manifold)
            for plot_fn in plot_fns:
                fig = plt.figure()
                try:
                    plot_fn(self)
                except SkipModel:
                    continue
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                data = np.swapaxes(data, 0, -1)
                data = np.swapaxes(data, -2, -1)

                name = plot_fn.__name__[len("plot_"):]
                self.logger.experiment.add_image(name, data, self.global_step)
                plt.close()

    def apply_conditions(self, batch) -> ConditionedBatch:
        x0 = batch[0]
        base_cond_shape = (x0.shape[0], 1)
        device = x0.device
        dtype = x0.dtype

        conds = []

        # Dataset condition
        if len(batch) != (2 if self.is_conditional() else 1):
            raise ValueError("You must pass a batch including conditions for each dataset condition")
        if len(batch) > 1:
            conds.append(batch[1])

        # SoftFlow
        noise = self.hparams.noise
        if isinstance(noise, list):
            min_noise, max_noise = noise
            if not self.training:
                max_noise = min_noise
            noise_scale = rand_log_uniform(
                max_noise, min_noise,
                shape=base_cond_shape, device=device, dtype=dtype
            )
            x = x0 + torch.randn_like(x0) * (10 ** noise_scale)
            conds.append(noise_scale)
        else:
            if noise > 0:
                x = x0 + torch.randn_like(x0) * noise
            else:
                x = x0

        # Loss weight aware
        loss_weights = defaultdict(float, self.hparams.loss_weights)
        for loss_key, loss_weight in self.hparams.loss_weights.items():
            if isinstance(loss_weight, list):
                min_weight, max_weight = loss_weight
                if not self.training:
                    # Per default, select the first value in the list
                    max_weight = min_weight
                weight_scale = rand_log_uniform(
                    min_weight, max_weight,
                    shape=base_cond_shape, device=device, dtype=dtype
                )
                loss_weights[loss_key] = (10 ** weight_scale).squeeze(1)
                conds.append(weight_scale)

        if len(conds) == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        else:
            c = torch.cat(conds, -1)
        return ConditionedBatch(x0, x, loss_weights, c)


def build_model(models, data_dim: int, cond_dim: int):
    models = deepcopy(models)
    model = Sequential()
    for model_spec in models:
        module_name, class_name = model_spec.pop("name").rsplit(".", 1)
        model_spec["data_dim"] = data_dim
        model_spec["cond_dim"] = cond_dim
        model.append(
            getattr(import_module(module_name), class_name)(model_spec)
        )
        data_dim = model_spec["latent_dim"]
    return model


def soft_heaviside(pos, start, stop):
    return max(0., min(
        1.,
        (pos - start)
        /
        (stop - start)
    ))


def reconstruction_loss(a, b):
    return torch.sum((a - b) ** 2, -1)


def rand_log_uniform(vmin, vmax, shape, device, dtype):
    vmin, vmax = map(log10, [vmin, vmax])
    return torch.rand(
        shape, device=device, dtype=dtype
    ) * (vmin - vmax) + vmax


def wasserstein2_distance_gaussian_approximation(x1, x2):
    # Returns the squared 2-Wasserstein distance between the Gaussian approximation of two datasets x1 and x2
    # 1. Calculate mean and covariance of x1 and x2
    # 2. Use fact that tr( ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) ) = sum(eigvals( ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) )) 
    # = sum(eigvals( cov1 cov2 )^(1/2))
    # 3. Return ||m1 - m2||^2 + tr( cov1 + cov2 - 2 ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) )
    m1 = x1.mean(0)
    m2 = x2.mean(0)
    cov1 = (x1 - m1[None]).T @ (x1 - m1[None]) / x1.shape[0]
    cov2 = (x2 - m2[None]).T @ (x2 - m2[None]) / x2.shape[0]
    cov_product = cov1 @ cov2
    eigenvalues_prod = torch.relu(torch.linalg.eigvals(cov_product).real)
    m_part = torch.sum((m1 - m2) ** 2)
    cov_part = torch.trace(cov1) + torch.trace(cov2) - 2 * torch.sum(torch.sqrt(eigenvalues_prod))
    return m_part + cov_part
