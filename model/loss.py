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
    assert x.size() == x_hat.size()
    return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x)


# def dmm_loss(x, x_hat, mu1, logvar1, mu2, logvar2,
#              kl_annealing_factor=1, mask=None):
def dmm_loss(x, x_hat, mask=None):
    # kl_raw = kl_div(mu1, logvar1, mu2, logvar2)
    nll_raw = nll_loss(x_hat, x)
    # feature-dimension reduced
    # kl_fr = kl_raw.mean(dim=-1)
    nll_fr = nll_raw.mean(dim=-1)
    # masking
    if mask is not None:
        mask = mask.gt(0).view(-1)
        # kl_m = kl_fr.view(-1).masked_select(mask).mean()
        nll_m = nll_fr.view(-1).masked_select(mask).mean()
    else:
        # kl_m = kl_fr.view(-1).mean()
        nll_m = nll_fr.view(-1).mean()

    # loss = kl_m * kl_annealing_factor + nll_m
    loss = nll_m

    # return kl_raw, nll_raw, \
    #     kl_fr, nll_fr, \
    #     kl_m, nll_m, \
    #     loss
    return loss
