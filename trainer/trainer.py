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
    def __init__(self, model, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None, overfit_single_batch=False):
        super().__init__(model, optimizer, config)
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

        self.img_log_interval = 50

    def _forward_step(self, batch, epoch, batch_idx, logger=None):
        x, x_reversed, x_mask, x_seq_lengths = batch

        x = x.to(self.device)
        x_reversed = x_reversed.to(self.device)
        x_mask = x_mask.to(self.device)
        x_seq_lengths = x_seq_lengths.to(self.device)

        # not all models need this, to be refractored
        kl_annealing_factor = \
            determine_annealing_factor(self.config['trainer']['min_anneal_factor'],
                                       self.config['trainer']['anneal_update'],
                                       epoch - 1, self.len_epoch, batch_idx)

        results = self.model(x, x_reversed, x_seq_lengths)
        losses = self.model.loss_function(*results,
                                          kl_weight=self.config['trainer']['kl_weight'],
                                          kl_annealing_factor=kl_annealing_factor,
                                          mask=x_mask)
        metrics = self.model.calculate_metrics(*results, mask=x_mask)

        if logger is not None:
            for k, v in losses.items():
                logger.update(k, v)
            for k, v in metrics.items():
                logger.update(k, v)
        else:
            logger = MetricTracker(*[l_i for l_i in losses.keys()],
                                   *[m_i for m_i in metrics.keys()],
                                   writer=self.writer)

        return results, losses['loss'], logger

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        logger = None
        # ----------------
        # add logging grad
        dict_grad = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and 'bias' not in name:
                dict_grad[name] = np.zeros(self.len_epoch)
        # ----------------

        for batch_idx, batch in enumerate(self.data_loader):

            results, loss, logger = self._forward_step(batch, epoch, batch_idx, logger)
            self.optimizer.zero_grad()
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
            for l_i in logger.item:
                logger.write_to_logger(l_i)
            # log gradients
            for name, p in dict_grad.items():
                self.writer.add_histogram(name + '/grad', p, bins='auto')
            # log parameters
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        # ---------------------------------------------------

        if epoch % self.img_log_interval == 0:
            fig = create_reconstruction_figure(results[1], torch.sigmoid(results[0]))
            self.writer.set_step(epoch, 'train')
            self.writer.add_figure('reconstruction', fig)

        log = logger.result()

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
        logger = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):

                results, loss, logger = self._forward_step(batch, epoch, batch_idx, logger)

        # ---------------------------------------------------
        if self.writer is not None:
            self.writer.set_step(epoch, 'valid')
            # log losses
            for l_i in logger.item:
                logger.write_to_logger(l_i)
        # ---------------------------------------------------

        if epoch % self.img_log_interval == 0:
            fig = create_reconstruction_figure(results[1], torch.sigmoid(results[0]))
            self.writer.set_step(epoch, 'valid')
            self.writer.add_figure('reconstruction', fig)

        return logger.result()

    def _test_epoch(self, epoch):
        self.model.eval()
        logger = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_data_loader):

                results, loss, logger = self._forward_step(batch, epoch, batch_idx, logger)

            n_sample = 3
            output_seq, z_p_seq, mu_p_seq, logvar_p_seq = self.model.generate(n_sample, 100)
            output_seq = torch.sigmoid(output_seq)
            plt.close()
            fig, ax = plt.subplots(n_sample, 1, figsize=(10, n_sample * 10))
            for i in range(n_sample):
                ax[i].imshow(output_seq[i].T.cpu().detach().numpy(), origin='lower')
            self.writer.set_step(epoch, 'test')
            self.writer.add_figure('generation', fig)

        return logger.result()

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
