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

import mlflow
from mlflow.models import infer_signature


def train_ivae(x_tr, x_val, u, N, num_hidden_layers, epochs=10000, batch_size=64,
        lr=0.01, a=100, b=1, c=0, d=10, gamma=0, validation_freq=3):
    st = time.time()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Executing script on: {}\n'.format(device))
    factor = gamma > 0

    # reshape data for iVAE format
    d_aux = u.shape[1]
    M = x_tr.shape[1]
    x_tr = x_tr.swapaxes(1, 2).reshape(-1, M)
    x_val = x_val.swapaxes(1, 2).reshape(-1, M)
    u_tr = jnp.tile(u, (x_tr.shape[0] // u.shape[0], 1))
    u_val = jnp.tile(u, (x_val.shape[0] // u.shape[0], 1))
    X = torch.from_dlpack(dlpack.to_dlpack(x_tr))
    X_val = torch.from_dlpack(dlpack.to_dlpack(x_val))
    U = torch.from_dlpack(dlpack.to_dlpack(u_tr))
    U_val = torch.from_dlpack(dlpack.to_dlpack(u_val))

    model = cleanIVAE(data_dim=M, latent_dim=N, aux_dim=d_aux, hidden_dim=M,
                      n_layers=num_hidden_layers+1, activation='xtanh',
                      slope=.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                     patience=0,
                                                     verbose=True)

    shuffle_key = jr.PRNGKey(9999)
    N_data = X.shape[0]
    num_full_minibatches, remainder = divmod(N_data, batch_size)
    num_minibatches = num_full_minibatches+bool(remainder)
    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0

        # validate every n epoch
        if epoch % validation_freq == 0:
            val_loss = 0

        shuffle_key, shuffkey = jr.split(shuffle_key)
        shuffle_idx = jr.permutation(shuffkey, jnp.arange(X.shape[0])).tolist()
        x_epoch = X.clone()[shuffle_idx]
        u_epoch = U.clone()[shuffle_idx]
        for it in range(num_minibatches):
            #print('Epoch {} - iteration {}/{}'.format(epoch, it, num_minibatches))
            model.train()
            x_it = x_epoch[it*batch_size:(it+1)*batch_size]
            u_it = u_epoch[it*batch_size:(it+1)*batch_size]
            x_it_val = X_val[it*batch_size:(it+1)*batch_size]
            u_it_val = U_val[it*batch_size:(it+1)*batch_size]

            # train model
            optimizer.zero_grad()
            loss, s_est_it = model.elbo(x_it, u_it, N_data, a=a, b=b, c=c, d=d)
            del s_est_it
            loss.backward(retain_graph=factor)
            optimizer.step()
            train_loss += loss.item()

            # validate at intervals
            if epoch % validation_freq == 0:
                model.eval()
                v_loss, _ = model.elbo(x_it_val, u_it_val, N_data, a=a, b=b, c=c, d=d)
                del _
                val_loss += v_loss.item()

        train_loss /= num_minibatches

        # log training loss every epoch
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        if epoch % validation_freq == 0:
            mlflow.log_metric('val_loss', val_loss/num_minibatches, step=epoch)

        if epoch % validation_freq == 0:
            print('==> Epoch {}/{}:\t'
                  'train loss: {:.6f}\t'
                  'validation loss: {:.6f}'.format(epoch, epochs, train_loss,
                                                   val_loss/num_minibatches))
        else:
            print('==> Epoch {}/{}:\t'
                  'train loss: {:.6f}'.format(epoch, epochs, train_loss))


        scheduler.step(train_loss)

    print('\ntotal runtime: {}'.format(time.time() - st))

    # perform inference on validation set
    model.eval()
    _, _, _, s_est_val, _ = model(X_val, U_val)
    return s_est_val
