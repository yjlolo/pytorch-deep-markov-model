# pytorch-deep-markov-model
Attempts to PyTorch re-implementatoin of 
- [Deep Markov Model](https://arxiv.org/abs/1609.09869) [Krishnan et al., AAAI 2017]
- [Factorial Deep Markov Model](https://groups.csail.mit.edu/sls/publications/2019/SameerKhurana_ICASSP-2019.pdf) [Khurana et al., ICASSP 2019]

**Note:** 
1. Need further fine-tuning to match the reported performance
2. Currently only supports JSB polyphonic music dataset

## Usage
Training the model
- DMM

    python train.py -c configs/dmm.json

- FDMM

    python train.py -c configs/fdmm.json


Add flag `-i` to specifically name the experiment that is to be saved under `saved/`

## References
0. Project template brought from the [pytorch-template](https://github.com/victoresque/pytorch-template)
1. The original DMM [source code](https://github.com/clinicalml/structuredinference/tree/master/expt-polyphonic-fast) in Theano
2. DMM PyTorch implementation in [Pyro](https://github.com/pyro-ppl/pyro/tree/dev/examples/dmm)
3. Another DMM PyTorch implementation by [@guxd](https://github.com/guxd/deepHMM)

## To-Do
- [ ] fine-tune to match the reported performances in the paper
- [ ] correct (if any) errors in metric calculation, `model/metric.py`
- [ ] optimize important sampling
