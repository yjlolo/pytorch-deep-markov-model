import torch
import torch.nn as nn
from model.modules import Emitter, Transition, Combiner, RnnEncoder
import data_loader.polyphonic_dataloader as poly
from data_loader.seq_util import seq_collate_fn
from base import BaseModel


class DeepMarkovModel(BaseModel):

    def __init__(self,
                 input_dim,
                 z_dim,
                 emission_dim,
                 transition_dim,
                 rnn_dim,
                 rnn_type,
                 orthogonal_init,
                 sample=True):
        super().__init__()
        # specify parameters from `config`
        # self.config = config
        # self.anneal_epoch = config['annealing_epochs']
        # self.anneal_update = config['annealing_updates']
        # self.min_anneal_factor = config['min_annealing_factor']
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.rnn_type = rnn_type
        self.orthogonal_init = orthogonal_init
        self.sample = sample
        # self.n_mini_batch = len(self.train_dataloader())

        # instantiate components of DMM
        # generative model
        self.emitter = Emitter(z_dim, emission_dim, input_dim)
        self.transition = Transition(z_dim, transition_dim,
                                     gated=True, identity_init=True)
        # inference model
        self.combiner = Combiner(z_dim, rnn_dim)
        self.encoder = RnnEncoder(input_dim, rnn_dim,
                                  n_layer=1, drop_rate=0.0,
                                  bd=False, nonlin='relu',
                                  rnn_type=rnn_type)

        # initialize hidden states
        # self.z_0 = self.transition.init_z_0()  # this does not seem to be updated during training
        self.mu_p_0, self.logvar_p_0 = self.transition.init_z_0()  # this does not seem to be updated during training
        self.z_q_0 = self.combiner.init_z_q_0()
        h_0 = self.encoder.init_hidden()
        if self.encoder.rnn_type == 'lstm':
            self.h_0, self.c_0 = h_0
        else:
            self.h_0 = h_0

        # if config['use_cuda']:
        #     self.cuda()

    def reparameterization(self, mu, logvar):
        if not self.sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def kl_div(self, mu1, logvar1, mu2=None, logvar2=None):
    #     if mu2 is None:
    #         mu2 = torch.zeros(1).to(mu1.device)
    #     if logvar2 is None:
    #         logvar2 = torch.zeros(1).to(mu1.device)

    #     return torch.sum(0.5 * (
    #         logvar2 - logvar1 + (
    #             torch.exp(logvar1) + (mu1-mu2).pow(2)
    #         ) / torch.exp(logvar2) - 1), -1)

    # def nll_loss(self, x_hat, x):
    #     return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x)

    # def forward(self, x, x_reversed, x_mask, x_seq_lengths):
    def forward(self, x, x_reversed, x_seq_lengths):
        T_max = x.size(1)
        batch_size = x.size(0)
        if self.encoder.rnn_type == 'lstm':
            h0 = self.h_0.expand(self.encoder.n_layer * self.encoder.n_direction,
                                 batch_size, self.rnn_dim).contiguous()
            c0 = self.c_0.expand(self.encoder.n_layer * self.encoder.n_direction,
                                 batch_size, self.rnn_dim).contiguous()
        else:
            h0 = self.h_0.expand(self.encoder.n_layer * self.encoder.n_direction,
                                 batch_size, self.rnn_dim).contiguous()
        # h_t carries information from t to T
        h_rnn = self.encoder(x_reversed, h0, x_seq_lengths)
        z_q_0 = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.mu_p_0.size(0))
        logvar_p_0 = self.logvar_p_0.expand(batch_size, 1, self.logvar_p_0.size(0))
        z_prev = z_q_0

        mu_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        logvar_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        mu_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        logvar_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        z_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        z_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        x_recon = torch.zeros([batch_size, T_max, self.input_dim]).to(x.device)
        # kl_seq = torch.zeros([batch_size, T_max]).to(x.device)
        # nll_seq = torch.zeros([batch_size, T_max]).to(x.device)
        for t in range(T_max):
            # q(z_t | z_{t-1}, x_{t:T})
            mu_q, logvar_q = self.combiner(z_prev, h_rnn[:, t, :])
            zt_q = self.reparameterization(mu_q, logvar_q)
            z_prev = zt_q
            # p(z_t | z_{t-1})
            mu_p, logvar_p = self.transition(z_prev)
            zt_p = self.reparameterization(mu_p, logvar_p)

            xt_recon = self.emitter(zt_q).contiguous()

            mu_q_seq[:, t, :] = mu_q
            logvar_q_seq[:, t, :] = logvar_q
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_q_seq[:, t, :] = zt_q
            z_p_seq[:, t, :] = zt_p
            x_recon[:, t, :] = xt_recon

            # kl_seq[:, t] = self.kl_div(mu_q, logvar_q, mu_p, logvar_p)
            # nll_seq[:, t] = self.nll_loss(
            #     xt_recon.reshape(-1), x[:, t, :].reshape(-1)).reshape(batch_size, -1).mean(-1)
            # assert not torch.isnan(nll_seq[:, t]).any()

        # include the kl loss at the 0-th time-step (?) this should have impacts on learning z_0?
        # apply mask and sum over time-axis
        # nll_seq *= x_mask
        # kl_seq *= x_mask
        # nll_seq = nll_seq.sum(dim=-1)
        # kl_seq = kl_seq.sum(dim=-1)
        # return nll_seq.mean(), kl_seq.mean(), x_recon
        mu_p_seq = torch.cat([mu_p_0, mu_p_seq[:, :-1, :]], dim=1)
        logvar_p_seq = torch.cat([logvar_p_0, logvar_p_seq[:, :-1, :]], dim=1)
        z_p_0 = self.reparameterization(mu_p_0, logvar_p_0)
        z_p_seq = torch.cat([z_p_0, z_p_seq[:, :-1, :]], dim=1)

        return x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq

    # def determine_annealing_factor(self, epoch, batch_idx):
    #     n_updates = epoch * self.n_mini_batch + batch_idx + 1
    #     # if self.anneal_epoch > 0 and epoch < self.anneal_epoch:
    #     #     anneal_factor = self.min_anneal_factor + \
    #     #         (1.0 - self.min_anneal_factor) * (
    #     #             (float(batch_idx + epoch * self.n_mini_batch + 1) /
    #     #                  float(self.anneal_epoch * self.n_mini_batch))
    #     #         )
    #     if self.anneal_update > 0 and n_updates < self.anneal_update:
    #         anneal_factor = self.min_anneal_factor + \
    #             (1.0 - self.min_anneal_factor) * (
    #                 (n_updates / self.anneal_update)
    #             )
    #     else:
    #         anneal_factor = 1.0
    #     return anneal_factor

    # def training_step(self, batch, batch_idx):
    #     x, x_reversed, x_mask, x_seq_lengths = batch
    #     nll_loss, kl_loss, x_recon = self(x, x_reversed, x_mask, x_seq_lengths)
    #     assert not torch.isnan(x_recon).any()
    #     assert not torch.isnan(nll_loss)
    #     assert not torch.isnan(kl_loss)

    #     anneal_factor = self.determine_annealing_factor(self.current_epoch,
    #                                                     batch_idx)

    #     loss = nll_loss + anneal_factor * kl_loss
    #     result = pl.TrainResult(minimize=loss)
    #     result.log_dict({
    #         'loss': loss,
    #         'nll_loss': nll_loss,
    #         'kl_loss': kl_loss,
    #         'af': torch.tensor(anneal_factor)
    #     }, on_step=False, on_epoch=True)

    #     return result

    # def on_after_backward(self):
    #     self.log_parameter(log_grad=True)

    # def validation_step(self, batch, batch_idx):
    #     x, x_reversed, x_mask, x_seq_lengths = batch
    #     nll_loss, kl_loss, x_recon = self(x, x_reversed, x_mask, x_seq_lengths)
    #     assert not torch.isnan(nll_loss)
    #     assert not torch.isnan(kl_loss)
    #     loss = nll_loss + kl_loss
    #     result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)
    #     result.log_dict({
    #         'val_loss': loss,
    #         'val_nll_loss': nll_loss,
    #         'val_kl_loss': kl_loss
    #     }, on_step=False, on_epoch=True)

    #     if self.current_epoch % 5 == 0:
    #         self.log_image(x[0], x_recon[0])

    #     return result

    # def train_dataloader(self):
    #     return DataLoader(poly.PolyDataset(poly.JSB_CHORALES, 'train'),
    #                       batch_size=self.config['batch_size'], shuffle=True,
    #                       collate_fn=seq_collate_fn, num_workers=12)

    # def val_dataloader(self):
    #     return DataLoader(poly.PolyDataset(poly.JSB_CHORALES, 'valid'),
    #                       batch_size=self.config['batch_size'], shuffle=False,
    #                       collate_fn=seq_collate_fn, num_workers=12)

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.config['lr'],
    #                             betas=(self.config['beta1'],
    #                                    self.config['beta2']),
    #                             weight_decay=self.config['weight_decay']
    #                             )

    # def log_image(self, x, x_recon):
    #     x = x.cpu().detach().numpy()
    #     x_recon = x_recon.cpu().detach().numpy()
    #     fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 20))
    #     ax[0].imshow(x.T, origin='lower')
    #     ax[1].imshow(x_recon.T, origin='lower')
    #     # path = '/'.join(['reconstructions', str(self.current_epoch) + '.jpg'])
    #     # fig.savefig(path)
    #     self.logger.experiment.add_figure('reconstruction', fig)
    #     plt.close()

    # def log_parameter(self, log_grad=True, debug=True):
    #     for name, p in self.named_parameters():
    #         self.logger.experiment.add_histogram(name + '/weight', p, bins='auto')
    #         if log_grad and p.requires_grad:
    #             if p.grad is not None:
    #                 self.logger.experiment.add_histogram(name + '/grad', p.grad, bins='auto')
    #             else:
    #                 self.logger.experiment.add_histogram(name + '/grad_none', 999, bins='auto')

def main(config):
    mini_batch_size = config['batch_size']
    use_cuda = config['use_cuda']

    dmm = DeepMarkovModel(config)
    dl = dmm.train_dataloader()
    print(next(iter(dl)))
    print('done')

    # data = poly.load_data(poly.JSB_CHORALES)
    # training_seq_lengths = data['train']['sequence_lengths']
    # training_data_sequences = data['train']['sequences']
    # test_seq_lengths = data['test']['sequence_lengths']
    # test_data_sequences = data['test']['sequences']
    # val_seq_lengths = data['valid']['sequence_lengths']
    # val_data_sequences = data['valid']['sequences']
    # N_train_data = len(training_seq_lengths)
    # N_train_time_slices = float(torch.sum(training_seq_lengths))
    # N_mini_batches = int(N_train_data / mini_batch_size +
    #                      int(N_train_data % mini_batch_size > 0))

    # # how often we do validation/test evaluation during training
    # val_test_frequency = 50
    # # the number of samples we use to do the evaluation
    # n_eval_samples = 1
    # def rep(x):
    #     rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
    #     repeat_dims = [1] * len(x.size())
    #     repeat_dims[0] = n_eval_samples
    #     return x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape)

    # # get the validation/test data ready for the dmm: pack into sequences, etc.
    # val_seq_lengths = rep(val_seq_lengths)
    # test_seq_lengths = rep(test_seq_lengths)
    # val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = get_mini_batch(
    #     torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
    #     val_seq_lengths, cuda=use_cuda)
    # test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = get_mini_batch(
    #     torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
    #     test_seq_lengths, cuda=use_cuda)

    # dmm = DeepMarkovModel(config)

    # def process_minibatch(which_mini_batch, shuffled_indices):
    #     # compute which sequences in the training set we should grab
    #     mini_batch_start = (which_mini_batch * mini_batch_size)
    #     mini_batch_end = np.min([(which_mini_batch + 1) * mini_batch_size, N_train_data])
    #     mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
    #     # grab a fully prepped mini-batch using the helper function in the data loader
    #     mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
    #         = get_mini_batch(mini_batch_indices, training_data_sequences,
    #                               training_seq_lengths, cuda=use_cuda)

    #     return mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths

    # shuffled_indices = torch.randperm(N_train_data)
    # for which_mini_batch in range(N_mini_batches):
    #     mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths = process_minibatch(which_mini_batch, shuffled_indices)
    #     dmm(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)


if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')
    config = {
        'batch_size': 20,
        'use_cuda': True,
        'input_dim': 88,
        'z_dim': 100,
        'emission_dim': 100,
        'trans_dim': 200,
        'rnn_dim': 600,

        # 'lr': 0.0003,
        'lr': 0.0008,
        'beta1': 0.96,
        'beta2': 0.999,
        'clip_norm': 10.0,
        'lr_decay': 0.99996,
        'weight_decay': 2.0,

        'annealing_epochs': 1000,
        'annealing_updates': 5000,
        # 'min_annealing_factor': 0.1
        'min_annealing_factor': 0.0
    }
    trainer = pl.Trainer(gpus=1,
                         # track_grad_norm=2,
                         gradient_clip_val=config['clip_norm'],
                         logger=tb_logger,
                         fast_dev_run=False)
    model = DeepMarkovModel(config)

    trainer.fit(model)

    main(config)
