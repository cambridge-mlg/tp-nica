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


def set_up_mlflow(params_to_log, experiment_name):
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = cfg.experiments
    print('Running with ', cfg)

    # synthetic or real data experiment to run
    if cfg.experiment_name == 'synthetic':
        data = generate_synthetic()
    else:
        x_tr, x_te, t = get_cv4a_data(cfg.data_dir, cfg.experiment_name)

    # option to evaluate with iVAE as a baseline 
    if cfg.ivae.ivae_baseline == True:
        set_up_mlflow(cfg, "/"+cfg.experiment_name+"_iVAE_baseline")
        with mlflow.start_run():
            # log config
            mlflow.log_params(cfg)
            # train iVAE and perform inference on validation set
            s_val_est = train_ivae(x_tr=jnp.float32(x_tr),
                                   x_val=jnp.float32(x_te),
                                   u=jnp.float32(t),
                                   N=cfg.ivae.N,
                                   num_hidden_layers=cfg.ivae.L_est,
                                   epochs=cfg.ivae.num_epochs,
                                   batch_size=cfg.ivae.minib_size,
                                   lr=cfg.ivae.lr)


    #train(data, cfg)



if __name__ == "__main__":

    # run main 
    sys.exit(main())
