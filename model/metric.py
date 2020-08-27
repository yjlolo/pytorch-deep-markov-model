import torch
from torch.distributions import Normal
from model.loss import nll_loss, kl_div


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def nll_metric(output, target, mask):
    assert output.dim() == target.dim() == 3
    assert output.size() == target.size()
    assert mask.dim() == 2
    assert mask.size(1) == output.size(1)
    loss = nll_loss(output, target)  # (batch_size, time_step, input_dim)
    loss = mask * loss.sum(dim=-1)  # (batch_size, time_step)
    loss = loss.sum(dim=1, keepdim=True)  # (batch_size, 1)
    return loss


def kl_div_metric(output, target, mask):
    mu1, logvar1 = output
    mu2, logvar2 = target
    assert mu1.size() == mu2.size()
    assert logvar1.size() == logvar2.size()
    assert mu1.dim() == logvar1.dim() == 3
    assert mask.dim() == 2
    assert mask.size(1) == mu1.size(1)
    kl = kl_div(mu1, logvar1, mu2, logvar2)
    kl = mask * kl.sum(dim=-1)
    kl = kl.sum(dim=1, keepdim=True)
    return kl


def elbo_eval(output, target, mask):
    x_recon, mu_q, logvar_q = output
    x, mu_p, logvar_p = target
    # batch_size = x.size(0)
    neg_elbo = nll_metric(x_recon, x, mask) + \
        kl_div_metric([mu_q, logvar_q], [mu_p, logvar_p], mask)
    # tsbn_bound_sum = elbo.div(mask.sum(dim=1, keepdim=True)).sum().div(batch_size)
    bound_sum = neg_elbo.sum().div(mask.sum())
    return bound_sum


def importance_sample(model, x, x_reversed, x_seq_lengths, mask, n_sample=2000):
    x = x.repeat(n_sample, 1, 1)
    x_reversed = x_reversed.repeat(n_sample, 1, 1)
    x_seq_lengths = x_seq_lengths.repeat(n_sample)  # make sure the shape
    x_mask = mask.repeat(n_sample, 1)

    x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
        model(x, x_reversed, x_seq_lengths, enforce_sorted=False)

    q_dist = Normal(mu_q_seq, logvar_q_seq.exp().sqrt())  # make sure if the shape is eligible
    p_dist = Normal(mu_p_seq, logvar_p_seq.exp().sqrt())
    log_qz = q_dist.log_prob(z_q_seq).sum(dim=-1) * x_mask
    log_pz = p_dist.log_prob(z_q_seq).sum(dim=-1) * x_mask
    log_px_z = -1 * nll_loss(x_recon, x).sum(dim=-1) * x_mask
    ll_estimate = log_px_z.sum(dim=1, keepdim=True) + \
        log_pz.sum(dim=1, keepdim=True) - \
        log_qz.sum(dim=1, keepdim=True)
    ll_estimate = ll_estimate.sum().div(n_sample).div(mask.sum())
    return ll_estimate
