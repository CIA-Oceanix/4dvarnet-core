import hydra
from hydra.utils import get_class, instantiate, call
from main import FourDVarNetRunner
import hydra_config

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
        print(self.dataloaders['train'])
        self.time = self.cfg.test_dates
        self.setup(dm)


@hydra.main(config_path='hydra_config', config_name='main')
def main(cfg):
    dm = instantiate(cfg.datamodule)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    call(cfg.entrypoint, self=runner)


if __name__ == '__main__':
    main()
