# pytorch-deep-markov-model
PyTorch re-implementatoin of the Deep Markov Model (https://arxiv.org/abs/1609.09869)
```
@inproceedings{10.5555/3298483.3298543,
    author = {Krishnan, Rahul G. and Shalit, Uri and Sontag, David},
    title = {Structured Inference Networks for Nonlinear State Space Models},
    year = {2017},
    publisher = {AAAI Press},
    booktitle = {Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence},
    pages = {2101–2109},
    numpages = {9},
    location = {San Francisco, California, USA},
series = {AAAI'17}
}
```
**Note:** The calculated metrics in `model/metrics.py` do not match those reported in the paper, which might be (more likely) due to differences in parameter settings and metric calculations.

## Usage
Training the model with the default `config.json`:
    
    python train.py -c config.json


add `-i` flag to specifically name the experiment that is to be saved under `saved/`.

## `config.json`
This file specifies parameters and configurations.
Below explains some key parameters.

**A careful fine-tuning of the parameters seems necessary to match the reported performances.**
```javascript
{
    "arch": {
        "type": "DeepMarkovModel",
        "args": {
            "input_dim": 88,
            "z_dim": 100,
            "emission_dim": 100,
            "transition_dim": 200,
            "rnn_dim": 600,
            "rnn_type": "lstm",
            "rnn_layers": 1,
            "rnn_bidirection": false,
            "use_embedding": true,      // use extra linear layer before RNN
            "orthogonal_init": true,    // orthogonal initialization for RNN
	    "gated_transition": true,       // use linear/non-linear gated transition
            "train_init": false,        // make z0 trainble
            "mean_field": false,        // use mean-field posterior q(z_t | x)
            "reverse_rnn_input": true,  // condition z_t on future inputs i.e. q(z_t | x_r)
            "sample": true              // sample during reparameterization
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.0008,               // default value from the author's source code
            "weight_decay": 0.0,        // debugging stage indicates that 1.0 prevents training
            "amsgrad": true,
            "betas": [0.9, 0.999]
        }
    },
    "trainer": {
        "epochs": 3000,
        "overfit_single_batch": false,  // overfit one single batch for debug

        "save_dir": "saved/",
        "save_period": 500,
        "verbosity": 2,
        
        "monitor": "min val_loss",
        "early_stop": 100,

        "tensorboard": true,

        "min_anneal_factor": 0.0,
        "anneal_update": 5000
    }
}
```

## Acknowledgements
0. Project template brought from the [pytorch-template](https://github.com/victoresque/pytorch-template)
1. The original [source code](https://github.com/clinicalml/structuredinference/tree/master/expt-polyphonic-fast) in Theano
2. PyTorch implementation in [Pyro](https://github.com/pyro-ppl/pyro/tree/dev/examples/dmm) framework
3. Another PyTorch implementation by [@guxd](https://github.com/guxd/deepHMM)

## To-Do
- [ ] fine-tune to match the reported performances in the paper
- [ ] correct (if any) errors in metric calculation, `model/metric.py`
- [ ] optimize important sampling