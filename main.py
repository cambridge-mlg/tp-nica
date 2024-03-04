import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
print(jax.devices())

import os
os.environ["MPLCONFIGDIR"] = "/proj/herhal/.cache/"

import sys
import hydra
import mlflow
import pdb
from omegaconf import DictConfig

from cv4a_data import get_cv4a_data
from ivae import train_ivae


def set_up_mlflow(cfg):
    mlflow.set_tracking_uri("databricks")
    mlflow.start_experiment(cfg.experiment_name)
    mlflow.log_params(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    set_up_mlflow(cfg)

    cfg = cfg.experiments
    print('Running with ', cfg)

    if cfg.experiment_name == 'synthetic':
        data = generate_synthetic()
    else:
        x_tr, x_te, t = get_cv4a_data(cfg.data_dir, cfg.experiment_name)

    if cfg.ivae.ivae_baseline == True:
        s_features, ivae_loss_hist = train_ivae(X=jnp.float32(x_tr),
                                                u=jnp.float32(t),
                                                N=cfg.ivae.N,
                                                num_hidden_layers=cfg.ivae.L_est,
                                                epochs=cfg.ivae.num_epochs,
                                                batch_size=cfg.ivae.minib_size,
                                                lr=cfg.ivae.lr)

    pdb.set_trace()

    #train(data, cfg)



if __name__ == "__main__":

    # run main 
    sys.exit(main())
