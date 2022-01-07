import hydra
import pandas as pd
from datetime import datetime, timedelta
from hydra.utils import get_class, instantiate, call
from main import FourDVarNetRunner
from omegaconf import OmegaConf
import hydra_config
from pytorch_lightning import seed_everything
import numpy as np

class FourDVarNetHydraRunner(FourDVarNetRunner):
    def __init__(self, params, dm, lit_mod_cls):
        self.cfg = params
        self.filename_chkpt = self.cfg.ckpt_name
        self.dm = dm
        self.lit_cls = lit_mod_cls
        dm.setup()
        self.dataloaders = {
            'train': dm.train_dataloader(),
            'val': dm.val_dataloader(),
            'test': dm.test_dataloader(),
        }

        test_dates = np.concatenate([ \
                       [str(dt.date()) for dt in \
                       pd.date_range(dm.test_slices[i].start,dm.test_slices[i].stop)[(self.cfg.dT//2):-(self.cfg.dT//2)]] \
                      for i in range(len(dm.test_slices))])
        #print(test_dates)
        self.time = {'time_test' : test_dates}

        self.setup(dm)

@hydra.main(config_path='hydra_config', config_name='main')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=cfg.get('seed', None))
    dm = instantiate(cfg.datamodule)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    call(cfg.entrypoint, self=runner)

if __name__ == '__main__':
    main()
