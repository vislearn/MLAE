# Adapted from https://github.com/stefanradev93/BayesFlow/blob/Development/bayesflow/computational_utilities.py#L67
# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch

from scipy import stats
from sklearn.calibration import calibration_curve

MMD_BANDWIDTH_LIST = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]


def gaussian_kernel_matrix(x, y, sigmas=None):
    """Computes a Gaussian radial basis functions (RBFs) between the samples of x and y.

    We create a sum of multiple Gaussian kernels each having a width :math:`\\sigma_i`.

    Parameters
    ----------
    x       :  torch.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  torch.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    sigmas  : list(float), optional, default: None
        List which denotes the widths of each of the gaussians in the kernel.
        If `sigmas is None`, a default range will be used, contained in ``MMD_BANDWIDTH_LIST``

    Returns
    -------
    kernel  : torch.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
    """
    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    beta = 1.0 / (2.0 * (torch.unsqueeze(torch.tensor(sigmas).to(x), 1)))
    dist = torch.unsqueeze(torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1), dim=-1)
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))
    kernel = torch.reshape(torch.sum(torch.exp(-s), 0), (x.shape[0], y.shape[0]))
    return kernel


def inverse_multiquadratic_kernel_matrix(x, y, sigmas=None):
    """Computes an inverse multiquadratic RBF between the samples of x and y.

    We create a sum of multiple IM-RBF kernels each having a width :math:`\\sigma_i`.

    Parameters
    ----------
    x       :  torch.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  torch.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    sigmas  : list(float), optional, default: None
        List which denotes the widths of each of the gaussians in the kernel.
        If `sigmas is None`, a default range will be used, contained in `MMD_BANDWIDTH_LIST`

    Returns
    -------
    kernel  : torch.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
    """
    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    dist = torch.unsqueeze(torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1), dim=-1)
    sigmas = torch.unsqueeze(sigmas, 0)
    return torch.sum(sigmas / (dist + sigmas), dim=-1)


def mmd_kernel(x, y, kernel):
    """Computes the estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between random draws from
    the distributions `x ~ P` and `y ~ Q`.

    Parameters
    ----------
    x      : torch.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution `x ~ P`.
    y      : torch.Tensor of shape (M, num_features)
        An array of `M` random draws from the "target" distribution `y ~ Q`.
    kernel : callable
        A function which computes the distance between pairs of samples.

    Returns
    -------
    loss   : torch.Tensor of shape (,)
        The statistically biased squared maximum mean discrepancy (MMD) value.
    """
    loss = torch.mean(kernel(x, x), -1)
    loss += torch.mean(kernel(y, y), -1)
    loss -= 2 * torch.mean(kernel(x, y), -1)
    return loss


def mmd_kernel_unbiased(x, y, kernel):
    """Computes the unbiased estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions `x ~ P` and `y ~ Q`.

    Parameters
    ----------
    x      : torch.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution `x ~ P`.
    y      : torch.Tensor of shape (M, num_features)
        An array of `M` random draws from the "target" distribution `y ~ Q`.
    kernel : callable
        A function which computes the distance between pairs of random draws from `x` and `y`.

    Returns
    -------
    loss   : torch.Tensor of shape (,)
        The statistically unbiased squared maximum mean discrepancy (MMD) value.
    """

    m, n = x.shape[0], y.shape[0]
    loss = (1.0 / (m * (m - 1))) * torch.sum(kernel(x, x))
    loss += (1.0 / (n * (n - 1))) * torch.sum(kernel(y, y))
    loss -= (2.0 / (m * n)) * torch.sum(kernel(x, y))
    return loss


def expected_calibration_error(m_true, m_pred, num_bins=10):
    """Estimates the expected calibration error (ECE) of a model comparison network according to [1].

    [1] Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).
        Obtaining well calibrated probabilities using bayesian binning.
        In Proceedings of the AAAI conference on artificial intelligence (Vol. 29, No. 1).

    Important
    ---------
    Make sure that ``m_true`` are **one-hot encoded** classes!

    Parameters
    ----------
    m_true      : np.ndarray of shape (num_sim, num_models)
        The one-hot-encoded true model indices.
    m_pred      : tf.tensor of shape (num_sim, num_models)
        The predicted posterior model probabilities.
    num_bins    : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).

    Returns
    -------
    cal_errs    : list of length (num_models)
        The ECEs for each model.
    probs       : list of length (num_models)
        The bin information for constructing the calibration curves.
        Each list contains two arrays of length (num_bins) with the predicted and true probabilities for each bin.
    """

    # Convert tf.Tensors to numpy, if passed
    if type(m_true) is not np.ndarray:
        m_true = m_true.numpy()
    if type(m_pred) is not np.ndarray:
        m_pred = m_pred.numpy()

    # Extract number of models and prepare containers
    n_models = m_true.shape[1]
    cal_errs = []
    probs = []

    # Loop for each model and compute calibration errs per bin
    for k in range(n_models):
        y_true = (m_true.argmax(axis=1) == k).astype(np.float32)
        y_prob = m_pred[:, k]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=num_bins)

        # Compute ECE by weighting bin errors by bin size
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob)
        bin_total = np.bincount(binids, minlength=len(bins))
        nonzero = bin_total != 0
        cal_err = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

        cal_errs.append(cal_err)
        probs.append((prob_true, prob_pred))
    return cal_errs, probs


def maximum_mean_discrepancy(source_samples, target_samples, kernel="gaussian", mmd_weight=1.0, minimum=0.0):
    """Computes the MMD given a particular choice of kernel.

    For details, consult Gretton et al. (2012):
    https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    Parameters
    ----------
    source_samples : torch.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution.
    target_samples : torch.Tensor of shape  (M, num_features)
        An array of `M` random draws from the "target" distribution.
    kernel         : str in ('gaussian', 'inverse_multiquadratic'), optional, default: 'gaussian'
        The kernel to use for computing the distance between pairs of random draws.
    mmd_weight     : float, optional, default: 1.0
        The weight of the MMD value.
    minimum        : float, optional, default: 0.0
        The lower bound of the MMD value.

    Returns
    -------
    loss_value : tf.Tensor
        A scalar Maximum Mean Discrepancy, shape (,)
    """

    # Determine kernel, fall back to Gaussian if unknown string passed
    if kernel == "gaussian":
        kernel_fun = gaussian_kernel_matrix
    elif kernel == "inverse_multiquadratic":
        kernel_fun = inverse_multiquadratic_kernel_matrix
    else:
        kernel_fun = gaussian_kernel_matrix

    # Compute and return MMD value
    loss_value = mmd_kernel(source_samples, target_samples, kernel=kernel_fun)
    loss_value = mmd_weight * torch.maximum(torch.tensor(minimum), loss_value)
    return loss_value


def get_coverage_probs(z, u):
    """Vectorized function to compute the minimal coverage probability for uniform
    ECDFs given evaluation points z and a sample of samples u.

    Parameters
    ----------
    z  : np.ndarray of shape (num_points, )
        The vector of evaluation points.
    u  : np.ndarray of shape (num_simulations, num_samples)
        The matrix of simulated draws (samples) from U(0, 1)
    """

    N = u.shape[1]
    F_m = np.sum((z[:, np.newaxis] >= u[:, np.newaxis, :]), axis=-1) / u.shape[1]
    bin1 = stats.binom(N, z).cdf(N * F_m)
    bin2 = stats.binom(N, z).cdf(N * F_m - 1)
    gamma = 2 * np.min(np.min(np.stack([bin1, 1 - bin2], axis=-1), axis=-1), axis=-1)
    return gamma


def simultaneous_ecdf_bands(
        num_samples, num_points=None, num_simulations=1000, confidence=0.95, eps=1e-5, max_num_points=1000
):
    """Computes the simultaneous ECDF bands through simulation according to
    the algorithm described in Section 2.2:

    https://link.springer.com/content/pdf/10.1007/s11222-022-10090-6.pdf

    Depends on the vectorized utility function `get_coverage_probs(z, u)`.

    Parameters
    ----------
    num_samples     : int
        The sample size used for computing the ECDF. Will equal to the number of posterior
        samples when used for calibrarion. Corresponds to `N` in the paper above.
    num_points      : int, optional, default: None
        The number of evaluation points on the interval (0, 1). Defaults to `num_points = num_samples` if
        not explicitly specified. Correspond to `K` in the paper above.
    num_simulations : int, optional, default: 1000
        The number of samples of size `n_samples` to simulate for determining the simultaneous CIs.
    confidence      : float in (0, 1), optional, default: 0.95
        The confidence level, `confidence = 1 - alpha` specifies the width of the confidence interval.
    eps             : float, optional, default: 1e-5
        Small number to add to the lower and subtract from the upper bound of the interval [0, 1]
        to avoid edge artefacts. No need to touch this.
    max_num_points  : int, optional, default: 1000
        Upper bound on `num_points`. Saves computation time when `num_samples` is large.
    Returns
    -------
    (alpha, z, L, U) - tuple of scalar and three arrays of size (num_samples,)
                       containing the confidence level as well as
                       the evaluation points, the lower, and the upper confidence bands, respectively.
    """

    # Use shorter var names throughout
    N = num_samples
    if num_points is None:
        K = min(N, max_num_points)
    else:
        K = min(num_points, max_num_points)
    M = num_simulations

    # Specify evaluation points
    z = np.linspace(0 + eps, 1 - eps, K)

    # Simulate M samples of size N
    u = np.random.uniform(size=(M, N))

    # Get alpha
    alpha = 1 - confidence

    # Compute minimal coverage probabilities
    gammas = get_coverage_probs(z, u)

    # Use insights from paper to compute lower and upper confidence interval
    gamma = np.percentile(gammas, 100 * alpha)
    L = stats.binom(N, z).ppf(gamma / 2) / N
    U = stats.binom(N, z).ppf(1 - gamma / 2) / N
    return alpha, z, L, U


def mean_squared_error(x_true, x_pred):
    """Computes the mean squared error between a single true value and M estimates thereof.

    x_true      : np.ndarray
    true values, shape ()

    x_pred      : np.ndarray
    predicted values, shape (M, )
    """
    x_true = np.array(x_true)
    x_pred = np.array(x_pred)
    try:
        return np.mean((x_true[np.newaxis, :] - x_pred) ** 2)
    except IndexError:
        return np.mean((x_true - x_pred) ** 2)


def root_mean_squared_error(x_true, x_pred):
    """Computes the mean squared error between a single true value and M estimates thereof.
    x_true      : np.ndarray
    true values, shape ()

    x_pred      : np.ndarray
    predicted values, shape (M, )
    """

    mse = mean_squared_error(x_true=x_true, x_pred=x_pred)
    return np.sqrt(mse)


def aggregated_error(x_true, x_pred, inner_error_fun=root_mean_squared_error, outer_aggregation_fun=np.mean):
    """Computes the aggregated error between a vector of N true values and M estimates of each true value.

    x_true      : np.ndarray
    true values, shape (N)

    x_pred      : np.ndarray
    predicted values, shape (M, N)

    inner_error_fun: callable, default: root_mean_squared_error
    computes the error between one true value and M estimates thereof

    outer_aggregation_fun: callable, default: np.mean
    aggregates N errors to a single aggregated error value
    """

    x_true, x_pred = np.array(x_true), np.array(x_pred)

    N = x_pred.shape[0]
    if not N == x_true.shape[0]:
        raise ValueError(f"x_true (len={N}) and x_pred (len={x_true.shape[0]}) must have the same length")

    errors = np.array([inner_error_fun(x_true=x_true[i], x_pred=x_pred[i]) for i in range(N)])

    if not N == errors.shape[0]:
        raise ValueError(f"x_true (len={N}) and errors (len={errors.shape[0]}) must have the same length")

    return outer_aggregation_fun(errors)


def aggregated_rmse(x_true, x_pred):
    """
    Computes the aggregated RMSE for a matrix of predictions.

    Parameters
    ----------
    x_true      : np.ndarray
    true values, shape (N)

    x_pred      : np.ndarray
    predicted values, shape (M, N)

    Returns
    -------
    aggregated RMSE

    """
    return aggregated_error(x_true=x_true,
                            x_pred=x_pred,
                            inner_error_fun=root_mean_squared_error,
                            outer_aggregation_fun=np.mean)
