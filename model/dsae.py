import warnings
import torch
import torch.nn as nn
from base import BaseModel
from .loss import nll_loss, mse_loss, kl_div
from .metric import nll_metric, kl_div_metric
from .modules import Emitter, Transition, Combiner, RnnEncoder, RnnGlobalEncoder
from .dmm import DeepMarkovModel
from .loss import post_process_output
from data_loader.seq_util import pack_padded_seq


class DisentangledSeqAE(DeepMarkovModel):
    def __init__(
        self,
        y_dim=32,
        avg_pool_global_var=True,
        global_var_cond_infer=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.y_dim = y_dim
        self.avg_pool_global_var = avg_pool_global_var
        self.global_var_cond_infer = global_var_cond_infer

        # q(y|x)
        self.global_encoder = RnnGlobalEncoder(
            y_dim, self.rnn_input_dim, self.rnn_dim,
            n_layer=self.rnn_layers, drop_rate=0.0,
            bd=True, nonlin='relu',
            rnn_type=self.rnn_type,
            reverse_input=False, 
            average_pool=avg_pool_global_var
        )
        # p(x_t|z_t, y)
        self.emitter = Emitter(
            self.z_dim, self.emission_dim, self.input_dim, y_dim=self.y_dim
        )
        # p(z_t|z_{<=t})
        # TODO: maybe simplify a bit with only a LSTM
        self.transition_lstm = nn.LSTM(
            self.z_dim, self.transition_dim, self.rnn_layers, batch_first=True
        )
        self.transition_fc = nn.Linear(self.transition_dim, self.z_dim)
        self.transition = Transition(
            self.z_dim, self.transition_dim,
            gated=self.gated_transition, identity_init=True
        )
        # q(z_t|z_{<=t}, x, y)
        self.combiner_lstm = nn.LSTM(
            self.z_dim, self.transition_dim, self.rnn_layers, batch_first=True
        )
        self.combiner_fc = nn.Linear(self.transition_dim, self.z_dim)
        self.combiner = Combiner(
            self.z_dim, self.rnn_dim,
            mean_field=self.mean_field,
            global_var_cond_infer=global_var_cond_infer,
            y_dim=y_dim
        )
        # h_{>=t} = f(x_{>=t})
        self.encoder = RnnEncoder(
            self.rnn_input_dim, self.rnn_dim,
            n_layer=self.rnn_layers, drop_rate=0.0,
            bd=self.rnn_bidirection, nonlin='relu',
            rnn_type=self.rnn_type, reverse_input=self.reverse_rnn_input
        )
        # Initialize hidden states
        self.mu_p_0, self.logvar_p_0 = \
            self.transition.init_z_0(trainable=self.train_init)
        self.z_q_0 = self.combiner.init_z_q_0(trainable=self.train_init)

    def encode_global(self, input, mask):
        mu, logvar = self.global_encoder(input, mask)
        z = self.reparameterization(mu, logvar)
        return mu, logvar, z

    def encode_local(self, input, z_prev, y, h_q=None, c_q=None):
        if h_q is None:
            pre_z_q, (h_q, c_q) = \
                self.combiner_lstm(z_prev.unsqueeze(1))
        else:
            pre_z_q, (h_q, c_q) = \
                self.combiner_lstm(z_prev.unsqueeze(1), (h_q, c_q))
        pre_z_q = self.combiner_fc(pre_z_q.squeeze(1))

        mu, logvar = self.combiner(
            h_x=input, z_t_1=pre_z_q, y=y,
            rnn_bidirection=self.rnn_bidirection
        )
        z_t = self.reparameterization(mu, logvar)

        return mu, logvar, z_t, h_q, c_q

    def prior_transition(self, input, h_p=None, c_p=None):
        if h_p is None:
            pre_z_p, (h_p, c_p) = \
                self.transition_lstm(input.unsqueeze(1))
        else:
            pre_z_p, (h_p, c_p) = \
                self.transition_lstm(input.unsqueeze(1), (h_p, c_p))
        pre_z_p = self.transition_fc(pre_z_p.squeeze(1))

        mu, logvar = self.transition(pre_z_p)
        z_t = self.reparameterization(mu, logvar)
        return mu, logvar, z_t, h_p, c_p

    def decode(self, z_t, y):
        x_t_recon = self.emitter(torch.cat([z_t, y], dim=-1)).contiguous()
        return x_t_recon

    def forward(self, x, x_reversed, x_seq_lengths, x_mask=None):
        device = x.device
        T_max = x.size(1)
        batch_size = x.size(0)

        if self.encoder.reverse_input:
            input = x_reversed
        else:
            input = x
        if self.use_embedding:
            input = self.embedding(input)
        input = pack_padded_seq(input, x_seq_lengths)

        # Encode the global latent variable
        mu_y, logvar_y, y = self.encode_global(input, x_mask)

        # Encode information from the input sequence
        h_rnn = self.encoder(input, x_seq_lengths)

        # Configure intitial states
        z_q_0 = self.z_q_0.expand(batch_size, self.z_dim)
        mu_p_0 = self.mu_p_0.expand(batch_size, self.z_dim)
        logvar_p_0 = self.logvar_p_0.expand(batch_size, self.z_dim)
        z_prev = z_q_0

        # Create placeholders
        x_recon = torch.zeros([batch_size, T_max, self.input_dim]).to(device)
        mu_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(device)
        logvar_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(device)
        mu_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(device)
        logvar_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(device)
        z_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(device)
        z_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(device)
        h_q = None
        c_q = None
        h_p = None
        c_p = None

        for t in range(T_max):
            # ---------------------------------------------------
            # q(z_t | z_{<t}, x_{t:T}, y), use x_{t:T} instead of 
            # x_{1:T} to mimic the true posterior
            # ---------------------------------------------------
            # Model the dependency on z_{<t} with a LSTM
            mu_q, logvar_q, zt_q, h_q, c_q = \
                self.encode_local(h_rnn[:, t, :], z_prev, y, h_q, c_q)
            z_prev = zt_q

            # ----------------------------------------------------------------
            # p(z_t | z_{<t}), note that z_{<t} is sampled from q instead of p
            # ----------------------------------------------------------------
            # Ideally, `z_prev` already contains all past information;
            # but we will need a RNN in our generative model for generation,
            # so we train the `transition_lstm` here as well to collect all the
            # past information on z_t
            mu_p, logvar_p, zt_p, h_p, c_p = \
                self.prior_transition(z_prev, h_p, c_p)

            # p(x | z_t, y)
            xt_recon = self.decode(zt_q, y)

            mu_q_seq[:, t, :] = mu_q
            logvar_q_seq[:, t, :] = logvar_q
            z_q_seq[:, t, :] = zt_q
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p_seq[:, t, :] = zt_p
            x_recon[:, t, :] = xt_recon

        mu_p_seq = torch.cat([mu_p_0.unsqueeze(1), mu_p_seq[:, :-1, :]], dim=1)
        logvar_p_seq = torch.cat(
            [logvar_p_0.unsqueeze(1), logvar_p_seq[:, :-1, :]], dim=1
        )
        z_p_0 = self.reparameterization(mu_p_0, logvar_p_0)
        z_p_seq = torch.cat([z_p_0.unsqueeze(1), z_p_seq[:, :-1, :]], dim=1)

        return x_recon, x, \
            z_q_seq, z_p_seq, y, \
            mu_q_seq, mu_p_seq, mu_y, \
            logvar_q_seq, logvar_p_seq, logvar_y

    def generate(self, batch_size, seq_len):
        mu_p = self.mu_p_0.expand(batch_size, self.z_dim)
        logvar_p = self.logvar_p_0.expand(batch_size, self.z_dim)
        to_device = mu_p.device

        z_p_seq = torch.zeros([batch_size, seq_len, self.z_dim]).to(to_device)
        mu_p_seq = torch.zeros([batch_size, seq_len, self.z_dim]).to(to_device)
        logvar_p_seq = torch.zeros(
            [batch_size, seq_len, self.z_dim]
        ).to(to_device)
        output_seq = torch.zeros(
            [batch_size, seq_len, self.input_dim]
        ).to(to_device)

        # Sample the global latent variable for the sequence
        y = self.reparameterization(
            torch.zeros(batch_size, self.y_dim).to(to_device),
            torch.zeros(batch_size, self.y_dim).to(to_device)
        )

        for t in range(seq_len):
            # Sample the local latent variable 
            z_p = self.reparameterization(mu_p, logvar_p)

            # Sample the observation
            xt = self.decode(z_p, y)

            # Sample the transition
            if t == 0:
                pre_z_p, (h_p, q_p) = self.transition_lstm(z_p.unsqueeze(1))
            else:
                pre_z_p, (h_p, q_p) =\
                    self.transition_lstm(z_p.unsqueeze(1), (h_p, q_p))
            pre_z_p = self.transition_fc(pre_z_p.squeeze(1))
            mu_p, logvar_p = self.transition(pre_z_p)

            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p_seq[:, t, :] = z_p
            output_seq[:, t, :] = xt

        return output_seq, z_p_seq, mu_p_seq, logvar_p_seq

    def loss_function(self, *args, **kwargs):
        recons, inputs = args[0], args[1]
        mu_q, mu_p, mu_y = args[5], args[6], args[7]
        logvar_q, logvar_p, logvar_y = args[8], args[9], args[10]
        kl_weight = kwargs['kl_weight']
        kl_annealing_factor = kwargs['kl_annealing_factor']
        mask = kwargs['mask']
        recon_obj = kwargs['recon_obj']

        recons, recon_loss_f = post_process_output(recons, recon_obj)
        nll_fr = recon_loss_f(recons, inputs).mean(dim=-1)

        kl_z_fr = kl_div(mu_q, logvar_q, mu_p, logvar_p).mean(dim=-1)
        kl_y_fr = kl_div(mu_y, logvar_y).mean()

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
            'kl_weigh': kl_weight
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