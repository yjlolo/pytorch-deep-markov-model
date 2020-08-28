import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None, overfit_single_batch=False):
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
        self.test_data_loader = test_data_loader if not overfit_single_batch else None
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
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
            self.test_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns], writer=self.writer)

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
            x_mask = x_mask.to(self.device)
            x_seq_lengths = x_seq_lengths.to(self.device)

            self.optimizer.zero_grad()
            x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
                self.model(x, x_reversed, x_seq_lengths)
            kl_annealing_factor = \
                determine_annealing_factor(self.config['trainer']['min_anneal_factor'],
                                           self.config['trainer']['anneal_update'],
                                           epoch - 1, self.len_epoch, batch_idx)
            kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss = \
                self.criterion(x, x_recon, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq, kl_annealing_factor, x_mask)
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
                    if met.__name__ == 'bound_eval':
                        self.train_metrics.update(met.__name__,
                                                  met([x_recon, mu_q_seq, logvar_q_seq],
                                                      [x, mu_p_seq, logvar_p_seq], mask=x_mask))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch or self.overfit_single_batch:
                break

        # ---------------------------------------------------
        if self.writer is not None:
            self.writer.set_step(epoch, 'train')
            # log losses
            for l_i in self.log_loss:
                self.train_metrics.write_to_logger(l_i)
            # log metrics
            if self.metric_ftns is not None:
                if met.__name__ == 'bound_eval':
                    self.train_metrics.write_to_logger(met.__name__)
            # log gradients
            for name, p in dict_grad.items():
                self.writer.add_histogram(name + '/grad', p, bins='auto')
            # log parameters
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
            # log kl annealing factors
            self.writer.add_scalar('anneal_factor', kl_annealing_factor)
        # ---------------------------------------------------

        if epoch % 50 == 0:
            fig = create_reconstruction_figure(x, torch.sigmoid(x_recon))
            # debug_fig = create_debug_figure(x, x_reversed, x_mask)
            # debug_fig_loss = create_debug_loss_figure(kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, x_mask)
            self.writer.set_step(epoch, 'train')
            self.writer.add_figure('reconstruction', fig)
            # self.writer.add_figure('debug', debug_fig)
            # self.writer.add_figure('debug_loss', debug_fig_loss)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.do_test and epoch % 50 == 0:
            test_log = self._test_epoch(epoch)
            log.update(**{'test_' + k: v for k, v in test_log.items()})

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
                x_mask = x_mask.to(self.device)
                x_seq_lengths = x_seq_lengths.to(self.device)

                x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
                    self.model(x, x_reversed, x_seq_lengths)
                kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss = \
                    self.criterion(x, x_recon, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq, 1, x_mask)

                for l_i, l_i_val in zip(self.log_loss, [loss, nll_m, kl_m]):
                    self.valid_metrics.update(l_i, l_i_val.item())
                if self.metric_ftns is not None:
                    for met in self.metric_ftns:
                        if met.__name__ == 'bound_eval':
                            self.valid_metrics.update(met.__name__,
                                                      met([x_recon, mu_q_seq, logvar_q_seq],
                                                          [x, mu_p_seq, logvar_p_seq], mask=x_mask))

        # ---------------------------------------------------
        if self.writer is not None:
            self.writer.set_step(epoch, 'valid')
            for l_i in self.log_loss:
                self.valid_metrics.write_to_logger(l_i)
            if self.metric_ftns is not None:
                for met in self.metric_ftns:
                    if met.__name__ == 'bound_eval':
                        self.valid_metrics.write_to_logger(met.__name__)
        # ---------------------------------------------------

        if epoch % 10 == 0:
            x_recon = torch.nn.functional.sigmoid(x_recon.view(x.size(0), x.size(1), -1))
            fig = create_reconstruction_figure(x, x_recon)
            # debug_fig = create_debug_figure(x, x_reversed_unpack, x_mask)
            # debug_fig_loss = create_debug_loss_figure(kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, x_mask)
            self.writer.set_step(epoch, 'valid')
            self.writer.add_figure('reconstruction', fig)
            # self.writer.add_figure('debug', debug_fig)
            # self.writer.add_figure('debug_loss', debug_fig_loss)

        return self.valid_metrics.result()

    def _test_epoch(self, epoch):
        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_data_loader):
                x, x_reversed, x_mask, x_seq_lengths = batch

                x = x.to(self.device)
                x_reversed = x_reversed.to(self.device)
                x_mask = x_mask.to(self.device)
                x_seq_lengths = x_seq_lengths.to(self.device)

                x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
                    self.model(x, x_reversed, x_seq_lengths)

                if self.metric_ftns is not None:
                    for met in self.metric_ftns:
                        if met.__name__ == 'bound_eval':
                            self.test_metrics.update(met.__name__,
                                                     met([x_recon, mu_q_seq, logvar_q_seq],
                                                         [x, mu_p_seq, logvar_p_seq], mask=x_mask))
                        if met.__name__ == 'importance_sample':
                            self.test_metrics.update(met.__name__,
                                                     met(batch_idx, self.model, x, x_reversed, x_seq_lengths, x_mask, n_sample=500))
        # ---------------------------------------------------
        if self.writer is not None:
            self.writer.set_step(epoch, 'test')
            if self.metric_ftns is not None:
                for met in self.metric_ftns:
                    self.test_metrics.write_to_logger(met.__name__)

            n_sample = 3
            output_seq, z_p_seq, mu_p_seq, logvar_p_seq = self.model.generate(n_sample, 100)
            output_seq = torch.sigmoid(output_seq)
            plt.close()
            fig, ax = plt.subplots(n_sample, 1, figsize=(10, n_sample * 10))
            for i in range(n_sample):
                ax[i].imshow(output_seq[i].T.cpu().detach().numpy(), origin='lower')
            self.writer.add_figure('generation', fig)
        # ---------------------------------------------------
        return self.test_metrics.result()

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
