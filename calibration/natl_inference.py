import torch
import hydra
from hydra.utils import get_class, instantiate, call
import hydra_config
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
import calibration
CKPT = (
    'dashboard/calmap_natl_2cascade/lightning_logs/version_2018995/checkpoints/'
    'modelCalSLAInterpGF-Exp3-epoch=171-val_loss=0.07.ckpt'.replace("=", r"\=")
)
xp = 'calmap_cascade_natl'



with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'entrypoint=test',
            f'entrypoint.ckpt_path={CKPT}'
        ]
    )
lit_cls = get_class(cfg.lit_mod_cls)
mod: calibration.lit_cal_model.LitCalModel = lit_cls.load_from_checkpoint(cfg.entrypoint.ckpt_path)
dm_cfg = cfg.datamodule
print(OmegaConf.to_yaml(dm_cfg))
dm = instantiate(dm_cfg)
dm.setup()
print('Done !')


trainer = pl.Trainer(gpus=1)
trainer.test(model=mod, dataloaders=dm.test_dataloader())
print('Done !')

