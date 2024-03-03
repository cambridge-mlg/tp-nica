import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
print(jax.devices())

import os
os.environ["MPLCONFIGDIR"] = "/proj/herhal/.cache/"

import sys
import hydra
import pdb
from omegaconf import DictConfig

from cv4a_data import get_cv4a_data
from ivae import train_ivae


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = cfg.experiments
    print('Running with ', cfg)

    if cfg.synthetic:
        data = generate_synthetic()
    else:
        x_tr, x_te, t = get_cv4a_data(cfg.data_dir, cfg.experiment_id)

    s_features, ivae_loss_hist = train_ivae(jnp.float32(x_tr),
                                            jnp.float32(t),
                                            cfg.ivae.N,
                                            cfg.ivae.L_est,
                                            cfg.ivae.num_epochs,
                                            cfg.ivae.minib_size)

    pdb.set_trace()

    #train(data, cfg)


if __name__ == "__main__":

    # parse args
    sys.exit(main())

    # run main
