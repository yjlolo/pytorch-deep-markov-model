import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from data_loader import seq_util


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, overfit_single_batch=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader if not overfit_single_batch else None
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.overfit_single_batch = overfit_single_batch

        # -------------------------------------------------
        # add flexibility to allow no metric in config.json
        self.log_loss = ['loss', 'nll', 'kl']
        if self.metric_ftns is None:
            self.train_metrics = MetricTracker(*self.log_loss, writer=self.writer)
            self.valid_metrics = MetricTracker(*self.log_loss, writer=self.writer)
        # -------------------------------------------------
        else:
            self.train_metrics = MetricTracker(*self.log_loss, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
            self.valid_metrics = MetricTracker(*self.log_loss, *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        # ----------------
        # add logging grad
        dict_grad = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and 'bias' not in name:
                dict_grad[name] = np.zeros(self.len_epoch)
        # ----------------

        for batch_idx, batch in enumerate(self.data_loader):
            x, x_reversed, x_mask, x_seq_lengths = batch

            x = x.to(self.device)
            x_reversed = x_reversed.to(self.device)
            x_pack = seq_util.pack_padded_seq(x, x_seq_lengths).to(self.device)
            x_reversed_pack = seq_util.pack_padded_seq(x_reversed, x_seq_lengths).to(self.device)
            x_mask = x_mask.to(self.device)
            x_seq_lengths = x_seq_lengths.to(self.device)

            self.optimizer.zero_grad()
            # x_recon, z_q_seq, mu_q_seq, logvar_q_seq = \
            x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
                self.model(x, x_reversed, x_pack, x_reversed_pack, x_seq_lengths)
            kl_annealing_factor = \
                determine_annealing_factor(self.config['trainer']['min_anneal_factor'],
                                           self.config['trainer']['anneal_update'],
                                           epoch - 1, self.len_epoch, batch_idx)
            # kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss = \
            #     self.criterion(x, x_recon, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq, 0, x_mask)
            kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss = \
                self.criterion(x, x_recon, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq, kl_annealing_factor, x_mask)
                #self.criterion(x, x_recon, mu_q_seq, logvar_q_seq, kl_annealing_factor, x_mask)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            # ------------
            # accumulate gradients that are to be logged later after epoch ends
            for name, p in self.model.named_parameters():
                if p.requires_grad and 'bias' not in name:
                    val = 0 if p.grad is None else p.grad.abs().mean()
                    dict_grad[name][batch_idx] = val
            # ------------

            self.optimizer.step()

            for l_i, l_i_val in zip(self.log_loss, [loss, nll_m, kl_m]):
                self.train_metrics.update(l_i, l_i_val.item())
            if self.metric_ftns is not None:
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            # ---------------------------------------------------
            # add flexibility to log either per step or per epoch
            if self.writer is not None:
                if not self.config['trainer']['log_on_epoch']:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    # for l_i, l_i_val in zip(self.log_loss, [loss, nll_aggr, kl_aggr]):
                    for l_i, l_i_val in zip(self.log_loss, [loss]):
                        self.train_metrics.write_to_logger(l_i, l_i_val.item())
                    if self.metric_ftns is not None:
                        for met in self.metric_ftns:
                            self.train_metrics.write_to_logger(met.__name__, met(output, target))
            # ---------------------------------------------------

            if batch_idx == self.len_epoch or self.overfit_single_batch:
                break

        # ---------------------------------------------------
        # add flexibility to log either per step or per epoch
        if self.writer is not None:
            if self.config['trainer']['log_on_epoch']:
                self.writer.set_step(epoch)
                for l_i in self.log_loss:
                    self.train_metrics.write_to_logger(l_i)
                if self.metric_ftns is not None:
                    self.train_metrics.write_to_logger(met.__name__)
                for name, p in dict_grad.items():
                    self.writer.add_histogram(name + '/grad', p, bins='auto')
                 # add histogram of model parameters to the tensorboard
                for name, p in self.model.named_parameters():
                    self.writer.add_histogram(name, p, bins='auto')
                # self.writer.add_scalar('anneal_factor', kl_annealing_factor)
        # ---------------------------------------------------
        if epoch % 10 == 0:
            x_recon = torch.nn.functional.sigmoid(x_recon.view(x.size(0), x.size(1), -1))
            fig = create_reconstruction_figure(x, x_recon)
            debug_fig = create_debug_figure(x, x_reversed, x_mask)
            # debug_fig_loss = create_debug_loss_figure(kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, x_mask)
            self.writer.set_step(epoch)
            self.writer.add_figure('reconstruction', fig)
            self.writer.add_figure('debug', debug_fig)
            # self.writer.add_figure('debug_loss', debug_fig_loss)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                x, x_reversed, x_mask, x_seq_lengths = batch

                x = x.to(self.device)
                x_reversed = x_reversed.to(self.device)
                x_pack = seq_util.pack_padded_seq(x, x_seq_lengths).to(self.device)
                x_reversed_pack = seq_util.pack_padded_seq(x_reversed, x_seq_lengths).to(self.device)
                x_mask = x_mask.to(self.device)
                x_seq_lengths = x_seq_lengths.to(self.device)

                self.optimizer.zero_grad()
                x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
                    self.model(x, x_reversed, x_pack, x_reversed_pack, x_seq_lengths)
                kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, kl_aggr, nll_aggr, loss = \
                    self.criterion(x, x_recon, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq, 1, x_mask)

                for l_i, l_i_val in zip(self.log_loss, [loss, nll_aggr, kl_aggr]):
                    self.valid_metrics.update(l_i, l_i_val.item())
                if self.metric_ftns is not None:
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output, target))

                # ---------------------------------------------------
                # add flexibility to log either per step or per epoch
                if self.writer is not None:
                    if not self.config['trainer']['log_on_epoch']:
                        self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                        for l_i, l_i_val in zip(self.log_loss, [loss, nll_aggr, kl_aggr]):
                            self.valid_metrics.write_to_logger(l_i, l_i_val.item())
                        if self.metric_ftns is not None:
                            for met in self.metric_ftns:
                                self.valid_metrics.write_to_logger(met.__name__, met(output, target))
                # ---------------------------------------------------

        # ---------------------------------------------------
        # add flexibility to log either per step or per epoch
        if self.writer is not None:
            if self.config['trainer']['log_on_epoch']:
                self.writer.set_step(epoch, 'valid')
                for l_i in self.log_loss:
                    self.valid_metrics.write_to_logger(l_i)
                if self.metric_ftns is not None:
                    for met in self.metric_ftns:
                        self.valid_metrics.write_to_logger(met.__name__)
        # ---------------------------------------------------

        if epoch % 10 == 0:
            fig = create_reconstruction_figure(x[0], torch.nn.Sigmoid()(x_recon[0]))
            # debug_fig = create_debug_figure(x, x_reversed_unpack, x_mask)
            # debug_fig_loss = create_debug_loss_figure(kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, x_mask)
            self.writer.set_step(epoch, 'valid')
            self.writer.add_figure('reconstruction', fig)
            # self.writer.add_figure('debug', debug_fig)
            # self.writer.add_figure('debug_loss', debug_fig_loss)

       
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def determine_annealing_factor(min_anneal_factor,
                               anneal_update,
                               epoch, n_batch, batch_idx):
    n_updates = epoch * n_batch + batch_idx

    if anneal_update > 0 and n_updates < anneal_update:
        anneal_factor = min_anneal_factor + \
            (1.0 - min_anneal_factor) * (
                (n_updates / anneal_update)
            )
    else:
        anneal_factor = 1.0
    return anneal_factor


def create_reconstruction_figure(x, x_recon, sample=True):
    plt.close()
    print(x.shape, x_recon.shape)
    if sample:
        idx = np.random.choice(x.shape[0], 1)[0]
    else:
        idx = 0
    x = x[idx].cpu().detach().numpy()
    x_recon = x_recon[idx].cpu().detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 20))
    ax[0].imshow(x.T, origin='lower')
    ax[1].imshow(x_recon.T, origin='lower')
    return fig


def create_debug_figure(x, x_reversed_unpack, x_mask, sample=True):
    plt.close()
    if sample:
        idx = np.random.choice(x.shape[0], 1)[0]
    else:
        idx = 0
    x = x[idx].cpu().detach().numpy()
    x_reversed_unpack = x_reversed_unpack[idx].cpu().detach().numpy()
    x_mask = x_mask[idx].cpu().detach().numpy()
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 30))
    ax[0].imshow(x.T, origin='lower')
    ax[1].imshow(x_reversed_unpack.T, origin='lower')
    ax[2].imshow(np.tile(x_mask, (x.shape[0], 1)), origin='lower')
    return fig


def create_debug_loss_figure(kl_raw, nll_raw,
                             kl_fr, nll_fr,
                             kl_m, nll_m,
                             mask, sample=True):
    plt.close()
    if sample:
        idx = np.random.choice(kl_raw.shape[0], 1)[0]
    else:
        idx = 0
    mask = tensor2np(mask[idx])
    kl_raw, nll_raw = tensor2np(kl_raw[idx]), tensor2np(nll_raw[idx])  # (t, f)
    kl_fr, nll_fr = tensor2np(kl_fr[idx]), tensor2np(nll_fr[idx])  # (t, )
    kl_m, nll_m = tensor2np(kl_m[idx]), tensor2np(nll_m[idx])  # (t, )
    # kl_aggr, nll_aggr = tensor2np(kl_aggr[idx]), tensor2np(nll_aggr[idx])  # ()
    fig, ax = plt.subplots(4, 2, sharex=True, figsize=(20, 40))
    ax[0][0].imshow(kl_raw.T, origin='lower')
    ax[1][0].plot(kl_fr)
    ax[2][0].plot(kl_m)
    ax[3][0].plot(mask)
    ax[0][1].imshow(nll_raw.T, origin='lower')
    ax[1][1].plot(nll_fr)
    ax[2][1].plot(nll_m)
    ax[3][1].plot(mask)
    return fig


def tensor2np(t):
    return t.cpu().detach().numpy()
