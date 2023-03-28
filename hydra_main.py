import os


import hydra
import pandas as pd
from datetime import datetime, timedelta
from hydra.utils import get_class, instantiate, call
from omegaconf import OmegaConf
import hydra_config
import numpy as np


def _main(cfg):

    import pytorch_lightning as pl
    from runner import FourDVarNetHydraRunner
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(seed=cfg.get('seed', None))
    dm = instantiate(cfg.datamodule)
    if cfg.get('callbacks') is not None:
        callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    else:
        callbacks=[]

    if cfg.get('logger') is not None:
        print('instantiating logger')
        print(OmegaConf.to_yaml(cfg.logger))
        logger = instantiate(cfg.logger)
    else:
        logger=True
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls, callbacks=callbacks, logger=logger)
    call(cfg.entrypoint, self=runner)


main = hydra.main(config_path='hydra_config', config_name='main', version_base='1.2')(_main)

if __name__ == '__main__':
    main()
