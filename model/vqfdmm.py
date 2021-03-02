import warnings
import torch
import torch.nn as nn
from base import BaseModel
from .loss import nll_loss, mse_loss, kl_div, kl_div_cat, post_process_output
from .metric import nll_metric, kl_div_metric
from .modules import Emitter, Transition, Combiner, RnnEncoder, RnnGlobalEncoder
from .fdmm import FactorDeepMarkovModel 
from data_loader.seq_util import pack_padded_seq


class VqFactorDeepMarkovModel(FactorDeepMarkovModel):
    def __init__(
        self,
        n_codebook,
        sample_mean=False,  # ad-hoc way to disable Gausssan repar
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_codebook = n_codebook
        self.codebook = nn.Embedding(n_codebook, self.z_dim)
        self.codebook.weight.data.uniform_(-1 / n_codebook, 1 / n_codebook)

        self.z_enc_to_logit = nn.Linear(self.z_dim, n_codebook)

    def dist_to_logit(self, z_enc):
        # logit = torch.zeros((z_enc.size(0), self.n_codebook)).to(z_enc.device)
        # for c in range(self.n_codebook):
        #     logit[:, c] = \
        #         -0.5 * torch.abs(
        #             z_enc - self.codebook.weight[c].expand(*z_enc.size())
        #         ).pow(2).sum(dim=-1)
        # return logit
        return self.z_enc_to_logit(z_enc)

    def gumbel_repar(self, logit, temperature=0.5, hard=True):
        u = torch.rand(logit.size(), device=logit.device)
        u = - torch.log(- torch.log(u + 1e-20) + 1e-20)
        prob_gumbel = torch.softmax((logit + u) / temperature, dim=-1)

        if not hard:
            return prob_gumbel

        _, ind = prob_gumbel.max(dim=-1)
        prob_hard = torch.zeros_like(logit)
        prob_hard.scatter_(1, ind.view(-1, 1), 1)
        # set gradients w.r.t. `prob_hard` gradients w.r.t. `prob`
        prob_hard = (prob_hard - prob_gumbel).detach() + prob_gumbel

        return prob_hard

    def dense_to_quan(self, z_enc, temperature=0.5, hard=True):
        logit = self.dist_to_logit(z_enc)
        gumbel_sample = self.gumbel_repar(logit, temperature, hard)
        return logit, torch.matmul(gumbel_sample, self.codebook.weight)
        
    def forward(
            self, x, x_reversed, x_seq_lengths, x_mask=None, 
            hard=True, temperature=0.5
        ):
        T_max = x.size(1)
        batch_size = x.size(0)

        if self.encoder.reverse_input:
            input = x_reversed
        else:
            input = x

        if self.use_embedding:
            input = self.embedding(input)

        input = pack_padded_seq(input, x_seq_lengths)

        mu_y, logvar_y = self.global_encoder(input, x_mask)
        y = self.reparameterization(mu_y, logvar_y)

        h_rnn = self.encoder(input, x_seq_lengths)
        z_q_0 = self.z_q_0.expand(batch_size, self.z_dim)
        mu_p_0 = self.mu_p_0.expand(batch_size, self.z_dim)
        logit_p_0, z_p_0 = self.dense_to_quan(mu_p_0)
        _, z_prev = self.dense_to_quan(z_q_0)

        x_recon = torch.zeros([batch_size, T_max, self.input_dim]).to(x.device)
        logit_q_seq = torch.zeros([batch_size, T_max, self.n_codebook]).to(x.device)
        logit_p_seq = torch.zeros([batch_size, T_max, self.n_codebook]).to(x.device)
        z_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        z_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        for t in range(T_max):
            # q(z_t | z_{t-1}, x_{t:T})
            mu_q, _ = self.combiner(
                h_x=h_rnn[:, t, :],
                z_t_1=z_prev,
                y=y,
                rnn_bidirection=self.rnn_bidirection
            )
            logit_q, zt_q = self.dense_to_quan(mu_q, temperature, hard)
            z_prev = zt_q
            # p(z_t | z_{t-1})
            # TODO: might want to try p(z_t|z_{t-1}, y)
            mu_p, _ = self.transition(z_prev)  
            logit_p, zt_p = self.dense_to_quan(mu_p, temperature, hard)

            xt_recon = self.emitter(torch.cat([zt_q, y], dim=-1)).contiguous()

            logit_q_seq[:, t, :] = logit_q
            z_q_seq[:, t, :] = zt_q
            logit_p_seq[:, t, :] = logit_p
            z_p_seq[:, t, :] = zt_p
            x_recon[:, t, :] = xt_recon

        logit_p_seq = torch.cat([logit_p_0.unsqueeze(1), logit_p_seq[:, :-1, :]], dim=1)
        z_p_seq = torch.cat([z_p_0.unsqueeze(1), z_p_seq[:, :-1, :]], dim=1)

        return x_recon, x, \
            z_q_seq, z_p_seq, y, \
            logit_q_seq, logit_p_seq, mu_y, \
            logvar_y

    def generate(self, batch_size, seq_len):
        mu_p = self.mu_p_0.expand(batch_size, self.z_dim)
        logvar_p = self.logvar_p_0.expand(batch_size, self.z_dim)
        z_p_seq = torch.zeros(
            [batch_size, seq_len, self.z_dim]
        ).to(mu_p.device)
        mu_p_seq = torch.zeros(
            [batch_size, seq_len, self.z_dim]
        ).to(mu_p.device)
        logvar_p_seq = torch.zeros(
            [batch_size, seq_len, self.z_dim]
        ).to(mu_p.device)
        output_seq = torch.zeros(
            [batch_size, seq_len, self.input_dim]
        ).to(mu_p.device)

        y = self.reparameterization(
            torch.zeros(batch_size, self.y_dim).to(mu_p.device),
            torch.zeros(batch_size, self.y_dim).to(mu_p.device)
        )
        
        for t in range(seq_len):
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p = self.reparameterization(mu_p, logvar_p)

            xt = self.emitter(torch.cat([z_p, y], dim=-1))
            mu_p, logvar_p = self.transition(z_p)

            output_seq[:, t, :] = xt
            z_p_seq[:, t, :] = z_p
        return output_seq, z_p_seq, mu_p_seq, logvar_p_seq

    def loss_function(self, *args, **kwargs):
        recons, inputs = args[0], args[1]
        logit_q_seq, logit_p_seq = args[5], args[6]
        mu_y, logvar_y = args[7], args[8]
        kl_weight = kwargs['kl_weight']
        kl_annealing_factor = kwargs['kl_annealing_factor']
        mask = kwargs['mask']
        recon_obj = kwargs['recon_obj']

        if recon_obj == 'nll':
            nll_fr = nll_loss(torch.sigmoid(recons), inputs).mean(dim=-1)
        elif recon_obj == 'mse':
            nll_fr = mse_loss(torch.tanh(recons), inputs).mean(dim=-1)
        else:
            msg = "Specify `recon_obj` with ['nll', 'mse']."
            raise NotImplementedError(msg)

        kl_z_fr = kl_div_cat(logit_q_seq, logit_p_seq).mean(dim=-1)
        kl_y_fr = kl_div(mu_y, logvar_y).mean(dim=-1).mean()

        if mask is not None:
            mask = mask.gt(0).view(-1)
            kl_z_m = kl_z_fr.view(-1).masked_select(mask).mean()
            nll_m = nll_fr.view(-1).masked_select(mask).mean()
        else:
            kl_z_m = kl_z_fr.view(-1).mean()
            nll_m = nll_fr.view(-1).mean()
            
        kl_weight *= kl_annealing_factor
        loss = kl_weight * (kl_z_m + kl_y_fr) + nll_m

        return {
            'loss': loss,
            'recon_loss': nll_m,
            'kld': kl_z_m,
            'kld_y': kl_y_fr,
            'kl_weight': kl_weight
        }

    def calculate_metrics(self, *args, **kwargs):
        recons, inputs = args[0], args[1]
        logit_q_seq, logit_p_seq = args[5], args[6]
        mu_y, logvar_y = args[7], args[8]
        mask = kwargs['mask']
        recon_obj = kwargs['recon_obj']

        recons = post_process_output(recons, recon_obj)

        neg_elbo = nll_metric(recons, inputs, mask) + \
            kl_div_metric([mu_q, logvar_q], mask, target=[mu_p, logvar_p]) + \
            kl_div(mu_y, logvar_y).sum(dim=-1, keepdim=True)

        return {
            'bound': neg_elbo.sum().div(mask.sum())
        }
