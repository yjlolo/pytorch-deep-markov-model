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
    assert x_hat.dim() == x.dim() == 3
    assert x_hat.size(1) == x.size(1)
    assert x_hat.size(0) == x.size(0)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    T_max = x_hat.size(1)
    batch_size = x_hat.size(0)
    rec_loss = torch.zeros_like(x_hat)
    for t in range(T_max):
        rec_loss[:, t, :] = loss_fn(x_hat[:, t, :].view(-1), x[:, t, :].contiguous().view(-1)) \
            .view(batch_size, -1)
    # return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x)
    return rec_loss


def dmm_loss(kl_annealing_factor=1, mask=None, **kwargs):
    required_kwargs = sorted(['mu1', 'mu2', 'logvar1', 'logvar2', 'x', 'x_hat'])
    input_kwargs = sorted(list(kwargs.keys()))
    assert input_kwargs == required_kwargs
    mu1, logvar1 = kwargs['mu1'], kwargs['logvar1']
    mu2, logvar2 = kwargs['mu2'], kwargs['logvar2']
    x, x_hat = kwargs['x'], kwargs['x_hat']

    kl_raw = kl_div(mu1, logvar1, mu2, logvar2)  # .sum(dim=-1)
    nll_raw = nll_loss(x_hat, x)  # .sum(dim=-1)
    # feature-dimension reduced
    kl_fr = kl_raw.sum(dim=-1)
    nll_fr = nll_raw.sum(dim=-1)
    # masking
    if mask is not None:
        kl_m = mask * kl_fr
        nll_m = mask * nll_fr

    batch_size = x.size(0)
    # time- and batch- aggregated
    kl_aggr = kl_m.sum(dim=-1).sum(dim=0).div(batch_size)
    nll_aggr = nll_m.sum(dim=-1).sum(dim=0).div(batch_size)
    loss = kl_aggr * kl_annealing_factor + nll_aggr

    return kl_raw, nll_raw, \
        kl_fr, nll_fr, \
        kl_m, nll_m, \
        kl_aggr, nll_aggr, \
        loss
