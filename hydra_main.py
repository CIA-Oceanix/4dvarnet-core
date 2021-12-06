import hydra
import pandas as pd
from hydra.utils import get_class, instantiate, call
from main import FourDVarNetRunner
from omegaconf import OmegaConf
import hydra_config
from pytorch_lightning import seed_everything

class FourDVarNetHydraRunner(FourDVarNetRunner):
    def __init__(self, params, dm, lit_mod_cls, callbacks=None):
        self.cfg = params
        self.filename_chkpt = self.cfg.ckpt_name
        self.callbacks = callbacks
        self.dm = dm
        self.lit_cls = lit_mod_cls
        dm.setup()
        self.dataloaders = {
            'train': dm.train_dataloader(),
            'val': dm.val_dataloader(),
            'test': dm.test_dataloader(),
        }

        test_dates = [str(dt.date()) for dt in \
            pd.date_range(dm.test_slices[0].start, dm.test_slices[0].stop)][self.cfg.dT // 2: -self.cfg.dT // 2 +1]
        self.time = {'time_test' : test_dates}

        self.setup(dm)


@hydra.main(config_path='hydra_config', config_name='main')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=cfg.get('seed', None))
    dm = instantiate(cfg.datamodule)
    if cfg.get('callbacks') is not None:
        callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    else:
        callbacks=[]
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls, callbacks=callbacks)
    call(cfg.entrypoint, self=runner)



if __name__ == '__main__':
    main()
