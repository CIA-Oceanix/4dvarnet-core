from distutils import config
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path='hydra_config', config_name='main')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    from train_test_4dvar_AT import _main
    return _main(cfg)


if __name__ == "__main__":
    main()