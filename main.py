import jax
from jax import config
config.update("jax_enable_x64", True)
print(jax.devices())

import sys
import hydra
import pdb
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = cfg.experiments
    print('Running with ', cfg)

    if cfg.synthetic:
        data = generate_synthetic()
    else:
        data = load(cfg.data_dir)

    train(data, cfg)


if __name__ == "__main__":

    # parse args
    sys.exit(main())

    # run main
