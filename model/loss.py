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


def nll_loss(x_hat, x, x_seq_len):
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    T_max = x_hat.size(1)
    batch_size = x_hat.size(0)
    rec_loss = torch.zeros_like(x_hat)
    x = x.type_as(x_hat)
    loss = torch.tensor(0.0, device=x.device)
    for i in range(batch_size):
        loss_i = torch.tensor(0.0, device=x.device)
        for t in range(x_seq_len[i]):
        # rec_loss[:, t, :] = loss_fn(x_hat[:, t, :].contiguous().view(-1), x[:, t, :].contiguous().view(-1)) \
            # .view(batch_size, -1)
            loss_i += loss_fn(x_hat[i, t, :], x[i, t, :]).mean()
        loss += loss_i.div(x_seq_len[i])
    loss = loss.div(batch_size)
    return loss
    # rec_loss = rec_loss.mean(dim=-1)

    # mask = mask.gt(0).view(-1)
    # return rec_loss.view(-1).masked_select(mask).mean()


def dmm_loss(x, x_hat, mu1, logvar1, mu2, logvar2, kl_annealing_factor=1, mask=None):
    batch_size = x.size(0)
    kl_raw = kl_div(mu1, logvar1, mu2, logvar2)
    nll_raw = nll_loss(x_hat, x)
    # feature-dimension reduced
    kl_fr = kl_raw.sum(dim=-1)
    nll_fr = nll_raw.sum(dim=-1)
    # masking
    if mask is not None:
        mask = mask.type_as(x_hat)
        mask = mask.gt(0).view(-1)
        kl_m = kl_fr.view(-1).masked_select(mask).mean()
        nll_m = nll_fr.view(-1).masked_select(mask).mean()

    # time- and batch- aggregated
    kl_aggr = kl_m
    nll_aggr = nll_m
    # loss = kl_aggr * kl_annealing_factor + nll_aggr
    loss = nll_aggr

    return kl_raw, nll_raw, \
        kl_fr, nll_fr, \
        kl_m, nll_m, \
        kl_aggr, nll_aggr, \
        loss
