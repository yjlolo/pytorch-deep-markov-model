import torch
import torch.nn as nn
from model.modules import Emitter, Transition, Combiner, RnnEncoder
import data_loader.polyphonic_dataloader as poly
from data_loader.seq_util import seq_collate_fn, pack_padded_seq
from base import BaseModel


class DeepMarkovModel(BaseModel):

    def __init__(self,
                 input_dim,
                 z_dim,
                 emission_dim,
                 transition_dim,
                 rnn_dim,
                 rnn_type,
                 rnn_layers,
                 rnn_bidirection,
                 orthogonal_init,
                 use_embedding,
                 gated_transition,
                 train_init,
                 mean_field=False,
                 reverse_rnn_input=True,
                 sample=True):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.rnn_bidirection = rnn_bidirection
        self.use_embedding = use_embedding
        self.orthogonal_init = orthogonal_init
        self.gated_transition = gated_transition
        self.train_init = train_init
        self.mean_field = mean_field
        self.reverse_rnn_input = reverse_rnn_input
        self.sample = sample

        if use_embedding:
            self.embedding = nn.Linear(input_dim, rnn_dim)
            rnn_input_dim = rnn_dim
        else:
            rnn_input_dim = input_dim

        # instantiate components of DMM
        # generative model
        self.emitter = Emitter(z_dim, emission_dim, input_dim)
        self.transition = Transition(z_dim, transition_dim,
                                     gated=gated_transition, identity_init=True)
        # inference model
        self.combiner = Combiner(z_dim, rnn_dim, mean_field=mean_field)
        self.encoder = RnnEncoder(rnn_input_dim, rnn_dim,
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
        if not self.sample:
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
            mu_q, logvar_q = self.combiner(h_rnn[:, t, :], z_prev,
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

        return x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq

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
