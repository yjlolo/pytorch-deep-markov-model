import argparse
import warnings
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_train', module_data)
    try:
        valid_data_loader = config.init_obj('data_loader_valid', module_data)
    except Exception:
        warnings.warn("Validation dataloader not given.")
        valid_data_loader = None
    try:
        test_data_loader = config.init_obj('data_loader_test', module_data)
    except Exception:
        warnings.warn("Test dataloader not given.")
        test_data_loader = None

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    # try:
    #     metrics = [getattr(module_metric, met) for met in config['metrics']]
    # -------------------------------------------------
    # add flexibility to allow no metric in config.json
    # except Exception:
    #     warnings.warn("No metrics are configured.")
    #     metrics = None
    # -------------------------------------------------

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # -------------------------------------------------
    # add flexibility to allow no lr_scheduler in config.json
    try:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    except Exception:
        warnings.warn("No learning scheduler is configured.")
        lr_scheduler = None
    # -------------------------------------------------

    trainer = Trainer(model, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      overfit_single_batch=config['trainer']['overfit_single_batch'])

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--identifier', default=None, type=str,
                      help='unique identifier of the experiment (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
