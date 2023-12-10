import enum
import time
import pdb

import torch
import torch.nn.functional as F
from torch import detach, optim
from torch.utils.data import DataLoader

import numpy as np

import jax.random as jr
import jax.numpy as jnp
from jax import dlpack
from .models import cleanIVAE


def train_ivae(x, u, N, num_hidden_layers, epochs=10000, batch_size=64, lr=0.01,
               a=100, b=1, c=0, d=10, gamma=0, num_samples_to_use=-1):
    st = time.time()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Executing script on: {}\n'.format(device))

    factor = gamma > 0

    # option to use only subset of data
    x = x[:num_samples_to_use]

    # reshape data for iVAE format
    d_aux = u.shape[1]
    M = x.shape[1]
    x_all = x.swapaxes(1, 2).reshape(-1, M)
    u_all = jnp.tile(u, (x.shape[0], 1))

    model = cleanIVAE(data_dim=M, latent_dim=N, aux_dim=d_aux, hidden_dim=M,
                      n_layers=num_hidden_layers+1, activation='xtanh',
                      slope=.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                     patience=0,
                                                     verbose=True)

    loss_hist = []
    shuffle_key = jr.PRNGKey(9999)
    N_data = x_all.shape[0]
    num_full_minibatches, remainder = divmod(N_data, batch_size)
    num_minibatches = num_full_minibatches+bool(remainder)
    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0

        shuffle_key, shuffkey = jr.split(shuffle_key)
        shuffle_idx = jr.permutation(shuffkey, jnp.arange(x_all.shape[0]))
        x_epoch = x_all.copy()[shuffle_idx]
        u_epoch = u_all.copy()[shuffle_idx]
        for it in range(num_minibatches):
            print('iteration {}/{}'.format(it, num_minibatches))
            model.train()
            x_it = x_epoch[it*batch_size:(it+1)*batch_size]
            u_it = u_epoch[it*batch_size:(it+1)*batch_size]

            # transfer to torch tensfor from jax
            x_it = torch.from_dlpack(dlpack.to_dlpack(x_it))
            u_it = torch.from_dlpack(dlpack.to_dlpack(u_it))

            # train model
            optimizer.zero_grad()
            loss, s_est_it = model.elbo(x_it, u_it, N_data, a=a, b=b, c=c, d=d)
            loss.backward(retain_graph=factor)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= num_minibatches
        loss_hist.append(train_loss)
        print('==> Epoch {}/{}:\t'
              'train loss: {:.6f}'.format(epoch, epochs, train_loss))
        scheduler.step(train_loss)

    print('\ntotal runtime: {}'.format(time.time() - st))

    # evaluate perf on full dataset
    model.eval()
    with torch.no_grad():
        X = torch.from_dlpack(dlpack.to_dlpack(x_all))
        U = torch.from_dlpack(dlpack.to_dlpack(u_all))
        _, _, _, S_est_all, _ = model(X, U)
    return S_est_all, loss_hist
