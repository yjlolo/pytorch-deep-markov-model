import time
import torch
from torch.distributions import Normal
from model.loss import nll_loss, kl_div


def nll_metric(output, target, mask):
    assert output.dim() == target.dim() == 3
    assert output.size() == target.size()
    assert mask.dim() == 2
    assert mask.size(1) == output.size(1)
    loss = nll_loss(output, target)  # (batch_size, time_step, input_dim)
    loss = mask * loss.sum(dim=-1)  # (batch_size, time_step)
    loss = loss.sum(dim=1, keepdim=True)  # (batch_size, 1)
    return loss


def kl_div_metric(output, mask, target=None):
    mu1, logvar1 = output
    assert mu1.dim() == logvar1.dim() == 3
    assert mask.dim() == 2
    assert mask.size(1) == mu1.size(1)
    if target is not None:
        mu2, logvar2 = target
        assert mu1.size() == mu2.size()
        assert logvar1.size() == logvar2.size()
        kl = kl_div(mu1, logvar1, mu2, logvar2)
    else:
        kl = kl_div(mu1, logvar1)

    kl = mask * kl.sum(dim=-1)
    kl = kl.sum(dim=1, keepdim=True)
    return kl


def bound_eval(output, target, mask):
    x_recon, mu_q, logvar_q, mu_y, logvar_y = output
    x, mu_p, logvar_p = target
    # batch_size = x.size(0)
    if mu_y is not None:
        neg_elbo = nll_metric(x_recon, x, mask) + \
            kl_div_metric([mu_q, logvar_q], mask, target=[mu_p, logvar_p]) + \
            kl_div_metric([mu_y, logvar_y], mask, target=None)
    else:
        neg_elbo = nll_metric(x_recon, x, mask) + \
            kl_div_metric([mu_q, logvar_q], mask, target=[mu_p, logvar_p])
    # tsbn_bound_sum = elbo.div(mask.sum(dim=1, keepdim=True)).sum().div(batch_size)
    bound_sum = neg_elbo.sum().div(mask.sum())
    return bound_sum


def importance_sample(batch_idx, model, x, x_reversed, x_seq_lengths, mask, n_sample=500):
    sample_batch_size = 25
    n_batch = n_sample // sample_batch_size
    sample_left = n_sample % sample_batch_size
    if sample_left == 0:
        n_loop = n_batch
    else:
        n_loop = n_batch + 1

    ll_estimate = torch.zeros(n_loop).to(x.device)

    start_time = time.time()
    for i in range(n_loop):
        if i < n_batch:
            n_repeats = sample_batch_size
        else:
            n_repeats = sample_left

        x_tile = x.repeat_interleave(repeats=n_repeats, dim=0)
        x_reversed_tile = x_reversed.repeat_interleave(repeats=n_repeats, dim=0)
        x_seq_lengths_tile = x_seq_lengths.repeat_interleave(repeats=n_repeats, dim=0)
        mask_tile = mask.repeat_interleave(repeats=n_repeats, dim=0)

        x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
            model(x_tile, x_reversed_tile, x_seq_lengths_tile)

        q_dist = Normal(mu_q_seq, logvar_q_seq.exp().sqrt())
        p_dist = Normal(mu_p_seq, logvar_p_seq.exp().sqrt())
        log_qz = q_dist.log_prob(z_q_seq).sum(dim=-1) * mask_tile
        log_pz = p_dist.log_prob(z_q_seq).sum(dim=-1) * mask_tile
        log_px_z = -1 * nll_loss(x_recon, x_tile).sum(dim=-1) * mask_tile
        ll_estimate_ = log_px_z.sum(dim=1, keepdim=True) + \
            log_pz.sum(dim=1, keepdim=True) - \
            log_qz.sum(dim=1, keepdim=True)

        ll_estimate[i] = ll_estimate_.sum().div(mask.sum())

    ll_estimate = ll_estimate.sum().div(n_sample)
    print("%s-th batch, importance sampling took %.4f seconds." % (batch_idx, time.time() - start_time))

    return ll_estimate
