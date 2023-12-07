from numbers import Number

import pdb

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


def _check_inputs(size, mu, v):
    """helper function to ensure inputs are compatible"""
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
        if v is None:
            v = torch.Tensor([1])
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))


def log_normal(x, mu=None, v=None, broadcast_size=False):
    """compute the log-pdf of a normal distribution with diagonal covariance"""
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v)
    else:
        mu, v = _check_inputs(x.size(), mu, v)
    assert mu.shape == v.shape
    return -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))


def log_laplace(x, mu, b, broadcast_size=False):
    """compute the log-pdf of a laplace distribution with diagonal covariance"""
    # b might not have batch_dimension. This case is handled by _check_inputs
    if broadcast_size:
        mu, b = _check_inputs(x.size(), mu, b)
    else:
        mu, b = _check_inputs(None, mu, b)
    return -torch.log(2 * b) - (x - mu).abs().div(b)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))
        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class cleanIVAE(nn.Module):
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3,
                 activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior params
        self.prior_mean = torch.zeros(1)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers,
                        activation=activation, slope=slope)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers,
                     activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers,
                     activation=activation, slope=slope)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers,
                        activation=activation, slope=slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def prior(self, u):
        logl = self.logl(u)
        return logl.exp()

    def forward(self, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z, l = self.forward(x, u)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps_cu = log_normal(z, None, l).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent),
                               v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1),
                                dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False)
                   - np.log(M * N)).sum(dim=-1)
        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i)
                 - d * (logqs_i - logps_cu)).mean()
        return elbo, z
