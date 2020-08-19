import torch
import torch.nn as nn


def kl_div(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None:
        mu2 = torch.zeros(1).to(mu1.device)
    if logvar2 is None:
        logvar2 = torch.zeros(1).to(mu1.device)

    return 0.5 * (
        logvar2 - logvar1 + (
            torch.exp(logvar1) + (mu1 - mu2).pow(2)
        ) / torch.exp(logvar2) - 1)


def nll_loss(x_hat, x):
    return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x)


def dmm_loss(kl_annealing_factor=1, mask=None, **kwargs):
    required_kwargs = sorted(['mu1', 'mu2', 'logvar1', 'logvar2', 'x', 'x_hat'])
    input_kwargs = sorted(list(kwargs.keys()))
    assert input_kwargs == required_kwargs
    mu1, logvar1 = kwargs['mu1'], kwargs['logvar1']
    mu2, logvar2 = kwargs['mu2'], kwargs['logvar2']
    x, x_hat = kwargs['x'], kwargs['x_hat']

    kl_l = kl_div(mu1, logvar1, mu2, logvar2).sum(dim=-1)
    nll_l = nll_loss(x_hat, x).sum(dim=-1)
    if mask is not None:
        kl_l = mask * kl_l
        nll_l = mask * nll_l

    batch_size = x.size(0)
    nll_l = nll_l.sum(dim=-1).sum(dim=0).div(batch_size)
    kl_l = kl_l.sum(dim=-1).sum(dim=0).div(batch_size)

    return nll_l, kl_l, nll_l + kl_l * kl_annealing_factor
