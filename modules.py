import torch
import torch.nn as nn
from util import reverse_sequence, pad_and_reverse

"""
Generative modules
"""


class Emitter(nn.Module):
    """
    Parameterize the Bernoulli observation likelihood `p(x_t | z_t)`

    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    emission_dim: int
        Dim. of emission hidden units
    input_dim: int
        Dim. of inputs

    Returns
    -------
        A valid probability that parameterizes the
        Bernoulli distribution `p(x_t | z_t)`
    """
    def __init__(self, z_dim, emission_dim, input_dim):
        super().__init__()
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim

        self.lin1 = nn.Linear(z_dim, emission_dim)
        self.lin2 = nn.Linear(emission_dim, emission_dim)
        self.lin3 = nn.Linear(emission_dim, input_dim)
        self.act = nn.ReLU()
        self.out = nn.Sigmoid()

    def forward(self, z_t):
        h1 = self.act(self.lin1(z_t))
        h2 = self.act(self.lin2(h1))
        return self.out(self.lin3(h2))


class Transition(nn.Module):
    """
    Parameterize the diagonal Gaussian latent transition probability
    `p(z_t | z_{t-1})`

    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    transition_dim: int
        Dim. of transition hidden units
    gated: bool
        Use the gated mechanism to consider both linearity and non-linearity
    identity_init: bool
        Initialize the linearity transform as an identity matrix;
        ignored if `gated == False`

    Returns
    -------
    mu: tensor (b, z_dim)
        Mean that parameterizes the Gaussian
    logvar: tensor (b, z_dim)
        Log-variance that parameterizes the Gaussian
    """
    def __init__(self, z_dim, transition_dim, gated=True, identity_init=True):
        super().__init__()
        self.z_dim = z_dim
        self.transition_dim = transition_dim
        self.gated = gated
        self.identity_init = identity_init

        # compute the corresponding mean parameterizing the Gaussian
        self.lin1p = nn.Linear(z_dim, transition_dim)
        self.lin2p = nn.Linear(transition_dim, z_dim)

        if gated:
            # compute the gain (gate) of non-linearity
            self.lin1 = nn.Linear(z_dim, transition_dim)
            self.lin2 = nn.Linear(transition_dim, z_dim)
            # compute the linearity part
            self.lin_n = nn.Linear(z_dim, z_dim)

        # compute the logvar
        self.lin_v = nn.Linear(z_dim, z_dim)

        if gated and identity_init:
            self.lin_n.weight.data = torch.eye(z_dim)
            self.lin_n.bias.data = torch.zeros(z_dim)

        self.act_weight = nn.Sigmoid()
        self.act = nn.ReLU()

    def init_z_0(self):
        return nn.Parameter(torch.zeros(self.z_dim))

    def forward(self, z_t_1):
        _mu = self.act(self.lin1p(z_t_1))
        mu = self.lin2p(_mu)
        logvar = self.lin_v(self.act(mu))

        if self.gated:
            _gain = self.act(self.lin1(z_t_1))
            gain = self.act_weight(self.lin2(_gain))
            mu = (1 - gain) * self.lin_n(z_t_1) + gain * mu

        return mu, logvar


"""
Inference modules
"""


class Combiner(nn.Module):
    """
    Parameterize variational distribution `q(z_t | z_{t-1}, x_{t:T})`
    a diagonal Gaussian distribution

    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    rnn_dim: int
        Dim. of RNN hidden states

    Returns
    -------
    mu: tensor (b, z_dim)
        Mean that parameterizes the variational Gaussian distribution
    logvar: tensor (b, z_dim)
        Log-var that parameterizes the variational Gaussian distribution
    """
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim

        self.lin1 = nn.Linear(z_dim, rnn_dim)
        self.lin2 = nn.Linear(rnn_dim, z_dim)
        self.lin_v = nn.Linear(rnn_dim, z_dim)
        self.act = nn.Tanh()

    def init_z_q_0(self):
        return nn.Parameter(torch.zeros(self.z_dim))

    def forward(self, z_t_1, h_rnn):
        """
        z_t_1: tensor (b, z_dim)
        h_rnn: tensor (b, rnn_dim)
        """
        h_comb = 0.5 * (self.act(self.lin1(z_t_1)) + h_rnn)
        mu = self.lin2(h_comb)
        logvar = self.lin_v(h_comb)

        return mu, logvar


class RnnEncoder(nn.Module):
    """
    RNN encoder that outputs hidden states h_t using x_{t:T}

    Parameters
    ----------
    input_dim: int
        Dim. of inputs
    rnn_dim: int
        Dim. of RNN hidden states
    n_layer: int
        Number of layers of RNN
    drop_rate: float [0.0, 1.0]
        RNN dropout rate between layers
    bd: bool
        Use bi-directional RNN or not

    Returns
    -------
    h_rnn: tensor (b, T_max, rnn_dim * n_direction)
        RNN hidden states at every time-step
    """
    def __init__(self, input_dim, rnn_dim, n_layer=1, drop_rate=0.0, bd=False,
                 nonlin='relu'):
        super().__init__()
        self.n_direction = 1 if not bd else 2
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.drop_rate = drop_rate
        self.bd = bd
        self.nonlin = nonlin

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim,
                          nonlinearity=nonlin, batch_first=True,
                          bidirectional=bd, num_layers=n_layer,
                          dropout=drop_rate)

    def calculate_effect_dim(self):
        return self.rnn_dim * self.n_direction

    def init_hidden(self):
        return nn.Parameter(
            torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim)
            )

    def forward(self, x, h0, seq_lengths):
        """
        x: pytorch packed object
            input packed data; this can be obtained from
            `util.get_mini_batch()`
        h0: tensor (n_layer * n_direction, b, rnn_dim)
        seq_lengths: tensor (b, )
        """
        _h_rnn, _ = self.rnn(x, h0)
        h_rnn = pad_and_reverse(_h_rnn, seq_lengths)
        # may have to reshape to (b, T_max, rnn_dim * n_direction)
        return h_rnn
