import warnings
import torch
import torch.nn as nn
from base import BaseModel
from .loss import nll_loss, kl_div
from .metric import nll_metric, kl_div_metric
from .modules import Emitter, Transition, Combiner, RnnEncoder, RnnGlobalEncoder
from .dmm import DeepMarkovModel
from data_loader.seq_util import pack_padded_seq


class FactorDeepMarkovModel(DeepMarkovModel):
    # https://groups.csail.mit.edu/sls/publications/2019/SameerKhurana_ICASSP-2019.pdf
    def __init__(self,
                 y_dim=32,
                 avg_pool_global_var=True,
                 global_var_cond_infer=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.y_dim = y_dim
        self.avg_pool_global_var = avg_pool_global_var
        self.global_var_cond_infer = global_var_cond_infer

        # encoder for global latent variables
        self.global_encoder = RnnGlobalEncoder(y_dim, self.rnn_input_dim, self.rnn_dim,
                                               n_layer=self.rnn_layers, drop_rate=0.0,
                                               bd=True, nonlin='relu',
                                               rnn_type=self.rnn_type,
                                               reverse_input=False,
                                               average_pool=avg_pool_global_var)
        # generative model
        self.emitter = Emitter(self.z_dim, self.emission_dim, self.input_dim,
                               y_dim=self.y_dim)
        self.transition = Transition(self.z_dim, self.transition_dim,
                                     gated=self.gated_transition, identity_init=True)
        # inference model
        self.combiner = Combiner(self.z_dim, self.rnn_dim,
                                 mean_field=self.mean_field,
                                 global_var_cond_infer=global_var_cond_infer,
                                 y_dim=y_dim)
        self.encoder = RnnEncoder(self.rnn_input_dim, self.rnn_dim,
                                  n_layer=self.rnn_layers, drop_rate=0.0,
                                  bd=self.rnn_bidirection, nonlin='relu',
                                  rnn_type=self.rnn_type,
                                  reverse_input=self.reverse_rnn_input)

        # initialize hidden states
        self.mu_p_0, self.logvar_p_0 = self.transition.init_z_0(trainable=self.train_init)
        self.z_q_0 = self.combiner.init_z_q_0(trainable=self.train_init)

    def forward(self, x, x_reversed, x_seq_lengths, x_mask=None):
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
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.z_dim)
        logvar_p_0 = self.logvar_p_0.expand(batch_size, 1, self.z_dim)
        z_prev = z_q_0

        x_recon = torch.zeros([batch_size, T_max, self.input_dim]).to(x.device)
        mu_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        logvar_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        mu_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        logvar_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        z_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        z_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        for t in range(T_max):
            # q(z_t | z_{t-1}, x_{t:T})
            mu_q, logvar_q = self.combiner(h_x=h_rnn[:, t, :],
                                           z_t_1=z_prev,
                                           y=y,
                                           rnn_bidirection=self.rnn_bidirection)
            zt_q = self.reparameterization(mu_q, logvar_q)
            z_prev = zt_q
            # p(z_t | z_{t-1})
            mu_p, logvar_p = self.transition(z_prev)  # we can also make it p(z_t|z_{t-1}, y)
            zt_p = self.reparameterization(mu_p, logvar_p)

            xt_recon = self.emitter(torch.cat([zt_q, y], dim=-1)).contiguous()

            mu_q_seq[:, t, :] = mu_q
            logvar_q_seq[:, t, :] = logvar_q
            z_q_seq[:, t, :] = zt_q
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p_seq[:, t, :] = zt_p
            x_recon[:, t, :] = xt_recon

        mu_p_seq = torch.cat([mu_p_0, mu_p_seq[:, :-1, :]], dim=1)
        logvar_p_seq = torch.cat([logvar_p_0, logvar_p_seq[:, :-1, :]], dim=1)
        z_p_0 = self.reparameterization(mu_p_0, logvar_p_0)
        z_p_seq = torch.cat([z_p_0, z_p_seq[:, :-1, :]], dim=1)

        return x_recon, x, \
            z_q_seq, z_p_seq, y, \
            mu_q_seq, mu_p_seq, mu_y, \
            logvar_q_seq, logvar_p_seq, logvar_y

    def generate(self, batch_size, seq_len):
        mu_p = self.mu_p_0.expand(batch_size, self.z_dim)
        logvar_p = self.logvar_p_0.expand(batch_size, self.z_dim)
        z_p_seq = torch.zeros([batch_size, seq_len, self.z_dim]).to(mu_p.device)
        mu_p_seq = torch.zeros([batch_size, seq_len, self.z_dim]).to(mu_p.device)
        logvar_p_seq = torch.zeros([batch_size, seq_len, self.z_dim]).to(mu_p.device)
        output_seq = torch.zeros([batch_size, seq_len, self.input_dim]).to(mu_p.device)
        for t in range(seq_len):
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p = self.reparameterization(mu_p, logvar_p)

            y = self.reparameterization(torch.zeros(batch_size, self.y_dim).to(z_p.device),
                                        torch.zeros(batch_size, self.y_dim).to(z_p.device))

            xt = self.emitter(torch.cat([z_p, y], dim=-1))
            mu_p, logvar_p = self.transition(z_p)

            output_seq[:, t, :] = xt
            z_p_seq[:, t, :] = z_p
        return output_seq, z_p_seq, mu_p_seq, logvar_p_seq

    def loss_function(self, *args, **kwargs):
        recons, inputs = args[0], args[1]
        mu_q, mu_p, mu_y = args[5], args[6], args[7]
        logvar_q, logvar_p, logvar_y = args[8], args[9], args[10]
        kl_weight = kwargs['kl_weight']
        kl_annealing_factor = kwargs['kl_annealing_factor']
        mask = kwargs['mask']

        nll_fr = nll_loss(recons, inputs).mean(dim=-1)
        kl_z_fr = kl_div(mu_q, logvar_q, mu_p, logvar_p).mean(dim=-1)
        kl_y_fr = kl_div(mu_y, logvar_y).mean()

        if mask is not None:
            mask = mask.gt(0).view(-1)
            kl_z_m = kl_z_fr.view(-1).masked_select(mask).mean()
            nll_m = nll_fr.view(-1).masked_select(mask).mean()
        else:
            kl_z_m = kl_z_fr.view(-1).mean()
            nll_m = nll_fr.view(-1).mean()

        loss = kl_annealing_factor * kl_weight * (kl_z_m + kl_y_fr) + nll_m

        return {
            'loss': loss,
            'recon_loss': nll_m,
            'kld': kl_z_m,
            'kld_y': kl_y_fr,
            'kl_anneal': kl_annealing_factor
        }

    def calculate_metrics(self, *args, **kwargs):
        recons, inputs = args[0], args[1]
        mu_q, mu_p, mu_y = args[5], args[6], args[7]
        logvar_q, logvar_p, logvar_y = args[8], args[9], args[10]
        mask = kwargs['mask']

        neg_elbo = nll_metric(recons, inputs, mask) + \
            kl_div_metric([mu_q, logvar_q], mask, target=[mu_p, logvar_p]) + \
            kl_div(mu_y, logvar_y).sum(dim=-1, keepdim=True)

        return {
            'bound': neg_elbo.sum().div(mask.sum())
        }
