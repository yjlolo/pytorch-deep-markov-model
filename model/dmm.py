import warnings
import torch
import torch.nn as nn
from base import BaseModel
from .loss import nll_loss, kl_div
from .metric import nll_metric, kl_div_metric
from .modules import Emitter, Transition, Combiner, RnnEncoder, RnnGlobalEncoder
from data_loader.seq_util import pack_padded_seq


class DeepMarkovModel(BaseModel):
    # https://arxiv.org/pdf/1609.09869.pdf
    def __init__(self,
                 input_dim=88,
                 z_dim=100,
                 emission_dim=100,
                 transition_dim=200,
                 rnn_dim=600,
                 rnn_type='lstm',
                 rnn_layers=1,
                 rnn_bidirection=False,
                 orthogonal_init=True,
                 use_embedding=True,
                 gated_transition=True,
                 train_init=False,
                 mean_field=False,
                 reverse_rnn_input=True,
                 sample_mean=True):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.rnn_bidirection = rnn_bidirection
        self.orthogonal_init = orthogonal_init
        self.use_embedding = use_embedding
        self.gated_transition = gated_transition
        self.train_init = train_init
        self.mean_field = mean_field
        if rnn_bidirection and reverse_rnn_input:
            reverse_rnn_input = False
            warnings.warn("`rnn_bidirection==True`, set `reverse_rnn_input` to False to avoid confusion.")
        self.reverse_rnn_input = reverse_rnn_input
        self.sample_mean = sample_mean

        if use_embedding:
            self.embedding = nn.Linear(input_dim, rnn_dim)
            self.rnn_input_dim = rnn_dim
        else:
            self.rnn_input_dim = input_dim

        # generative model
        self.emitter = Emitter(z_dim, emission_dim, input_dim)
        self.transition = Transition(z_dim, transition_dim,
                                     gated=gated_transition, identity_init=True)
        # inference model
        self.combiner = Combiner(z_dim, rnn_dim,
                                 mean_field=mean_field)
        self.encoder = RnnEncoder(self.rnn_input_dim, rnn_dim,
                                  n_layer=rnn_layers, drop_rate=0.0,
                                  bd=rnn_bidirection, nonlin='relu',
                                  rnn_type=rnn_type,
                                  reverse_input=reverse_rnn_input)

        # initialize hidden states
        self.mu_p_0, self.logvar_p_0 = self.transition.init_z_0(trainable=train_init)
        self.z_q_0 = self.combiner.init_z_q_0(trainable=train_init)
        # h_0 = self.encoder.init_hidden(trainable=train_init)
        # if self.encoder.rnn_type == 'lstm':
        #     self.h_0, self.c_0 = h_0
        # else:
        #     self.h_0 = h_0

    def reparameterization(self, mu, logvar):
        if not self.sample_mean:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, x_reversed, x_seq_lengths):
        T_max = x.size(1)
        batch_size = x.size(0)

        if self.encoder.reverse_input:
            input = x_reversed
        else:
            input = x

        if self.use_embedding:
            input = self.embedding(input)

        input = pack_padded_seq(input, x_seq_lengths)

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
                                           rnn_bidirection=self.rnn_bidirection)
            zt_q = self.reparameterization(mu_q, logvar_q)
            z_prev = zt_q
            # p(z_t | z_{t-1})
            mu_p, logvar_p = self.transition(z_prev)
            zt_p = self.reparameterization(mu_p, logvar_p)

            xt_recon = self.emitter(zt_q).contiguous()

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
            z_q_seq, z_p_seq, \
            mu_q_seq, mu_p_seq, \
            logvar_q_seq, logvar_p_seq

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

            xt = self.emitter(z_p)
            mu_p, logvar_p = self.transition(z_p)

            output_seq[:, t, :] = xt
            z_p_seq[:, t, :] = z_p
        return output_seq, z_p_seq, mu_p_seq, logvar_p_seq

    def loss_function(self, *args, **kwargs):
        recons, inputs = args[0], args[1]
        mu_q, mu_p = args[4], args[5]
        logvar_q, logvar_p = args[6], args[7]
        kl_weight = kwargs['kl_weight']
        kl_annealing_factor = kwargs['kl_annealing_factor']
        mask = kwargs['mask']

        nll_fr = nll_loss(recons, inputs).mean(dim=-1)
        kl_fr = kl_div(mu_q, logvar_q, mu_p, logvar_p).mean(dim=-1)

        if mask is not None:
            mask = mask.gt(0).view(-1)
            kl_m = kl_fr.view(-1).masked_select(mask).mean()
            nll_m = nll_fr.view(-1).masked_select(mask).mean()
        else:
            kl_m = kl_fr.view(-1).mean()
            nll_m = nll_fr.view(-1).mean()

        loss = kl_annealing_factor * kl_weight * kl_m + nll_m

        return {
            'loss': loss,
            'recon_loss': nll_m,
            'kld': kl_m,
            'kl_anneal': kl_annealing_factor
        }

    def calculate_metrics(self, *args, **kwargs):
        recons, inputs = args[0], args[1]
        mu_q, mu_p = args[4], args[5]
        logvar_q, logvar_p = args[6], args[7]
        mask = kwargs['mask']

        neg_elbo = nll_metric(recons, inputs, mask) + \
            kl_div_metric([mu_q, logvar_q], mask, target=[mu_p, logvar_p])

        return {
            'bound': neg_elbo.sum().div(mask.sum())
        }
