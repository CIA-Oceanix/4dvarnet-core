# %% Load nad_sst model
import pytorch_lightning as pl
# from models import LitModel, LitModelWithSST
# from new_dataloading import FourDVarNetDataModule
# from main import FourDVarNetRunner
import main
ckpt_path= "first_results_dash/train/nad_roll/checkpoints/modelSLAInterpGF-Exp3-epoch=22-val_loss=0.07.ckpt"
config_pkg = 'q.nad_roll'
runner = main.FourDVarNetRunner(config=config_pkg)

mod = runner._get_model(ckpt_path=ckpt_path)

print(" #### ", config_pkg, " #### ")
# %% Generate maps

trainer = pl.Trainer(gpus=1)
trainer.test(mod, test_dataloaders=runner.dataloaders['test'])


# %%
from metrics import get_psd_score

fig, spatial_resolution_model, spatial_resolution_re = get_psd_score(mod.test_xr_ds.gt, mod.test_xr_ds.pred, mod.test_xr_ds.oi, with_fig=True)
