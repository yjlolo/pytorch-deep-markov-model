import torch
import torch.nn as nn


def post_process_output(output_seq, reconstruction_obj):
    if reconstruction_obj == 'nll':
        output_seq = torch.sigmoid(output_seq)
        loss_f = nll_loss
    elif reconstruction_obj == 'mse':
        # Currently only supports value range [-1, 1];
        # min-max normalisation has to be done accordingly
        output_seq = torch.tanh(output_seq)
        loss_f = mse_loss
    else:
        msg = "Specify `reconstruction_obj` with ['nll', 'mse']."
        raise NotImplementedError(msg)
    return output_seq, loss_f


def kl_div(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if logvar2 is None:
        logvar2 = torch.zeros_like(mu1)

    return 0.5 * (
        logvar2 - logvar1 + (
            torch.exp(logvar1) + (mu1 - mu2).pow(2)
        ) / torch.exp(logvar2) - 1)


def kl_div_cat(logit1, logit2=None):
    if logit2 is None:
        logit2 = torch.zeros_like(logit1)

    return torch.softmax(logit1, dim=-1) * \
        (torch.log_softmax(logit1, dim=-1) - torch.log_softmax(logit2, dim=-1))


def nll_loss(x_hat, x):
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x)


def mse_loss(x_hat, x):
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    return nn.MSELoss(reduction='none')(x_hat, x)
