import xarray as xr
from functools import reduce
import numpy as np
import sparse
import pandas as pd
import torch
import hydra
from hydra.utils import get_class, instantiate, call
import hydra_config
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
import calibration


xp = 'calmap_patch'
CKPT = 'archive_dash/finetune_calmap_gf_dec_lr/lightning_logs/version_2021874/checkpoints/modelCalSLAInterpGF-Exp3-epoch=49-val_loss=0.06.ckpt'
with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'entrypoint=train',
        ]
    )
import importlib
import new_dataloading
import calibration.lit_cal_model
import calibration.dataset
importlib.reload(new_dataloading)
importlib.reload(calibration.lit_cal_model)
importlib.reload(calibration.dataset)

dm = instantiate(cfg.datamodule)
dm.setup()
lit_mod = get_class(cfg.lit_mod_cls).load_from_checkpoint(CKPT, hparam=cfg.params)

trainer = pl.Trainer(gpus=1)
test_dl = dm.test_dataloader()
import inspect

test_ds = test_dl.dataset.datasets[0]
trainer.test(lit_mod, test_dataloaders=test_dl)
import matplotlib.pyplot as plt
lit_mod.test_xrds.sel(time='2013-01-16').gt.plot()

(lit_mod.test_xrds /lit_mod.test_xrds.weight).sel(time='2013-01-25').pred.plot()
isel(time=10).gt.plot()
lit_mod.test_xrds.isel(time=3).weight.plot()
plt.imshow(lit_mod.patch_weight[2, ...])
self = lit_mod
for i in self.outputs[0]['gt']:
    print(i.shape)

gt_dses = [
        xr.Dataset(
            {'gt': (('time', 'lat', 'lon'), gt_data)},
            coords=coords
        ).isel(time=self.hparams.dT//2)
        for gt_data, coords in zip(
           (gt_data for chunk in self.outputs  for gt_data in chunk['gt']),
            self.test_patch_coords
        )
]
len(self.outputs)
list_of_items = zip(*[chunk.values() for chunk in self.outputs])

def iter_item(outputs):
    for chunk in outputs:
        n = chunk['gt'].shape[0]
        for i in range(n):
            yield (
                    chunk['gt'][i],
                    chunk['oi'][i],
                    chunk['preds'][i],
                    chunk['target_obs'][i],
                    chunk['obs_pred'][i],
                    chunk['inp_obs'][i],
            )
    
dses =[ 
        xr.Dataset( {
            'gt': (('time', 'lat', 'lon'), x_gt),
            'oi': (('time', 'lat', 'lon'), x_oi),
            'pred': (('time', 'lat', 'lon'), x_rec),
            'obs_gt': (('time', 'lat', 'lon'), obs_gt),
            'obs_pred': (('time', 'lat', 'lon'), obs_pred),
            'obs_inp': (('time', 'lat', 'lon'), obs_inp),
        }, coords=coords)
        for  (x_gt, x_oi, x_rec, obs_gt, obs_pred, obs_inp), coords in zip(iter_item(self.outputs), self.test_patch_coords)
]


print(1)
from functools import reduce
import time

t0 = time.time()
fin_ds = reduce(lambda ds1, ds2: xr.merge([ds1, ds2]), [xr.zeros_like(ds[['lat', 'lon', 'time']]) for ds in dses])
fin_ds = xr.merge([xr.zeros_like(ds[['lat', 'lon', 'time']]) for ds in dses])
fin_ds = fin_ds.assign(
    {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
)
for v in dses[0]:
    fin_ds = fin_ds.assign(
        {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
    )
print(time.time() -t0)

patch_weight = np.zeros(tuple(cfg.datamodule.slice_win.values()), dtype='float32')
patch_weight[20:-20, 20:-20, 2:-2] = 1.

for ds in dses:
    ds_nans = ds.assign(weight=xr.ones_like(ds.gt)).isnull().broadcast_like(fin_ds).fillna(0.)
    # print(ds_nans)
    xr_weight =xr.DataArray(patch_weight, ds.coords, dims=tuple(cfg.datamodule.slice_win.keys())) 
    _ds = ds.pipe(lambda dds: dds * xr_weight).assign(weight=xr_weight).broadcast_like(fin_ds).fillna(0.).where(ds_nans==0, np.nan)
    # print(_ds)
    fin_ds = fin_ds + _ds 
    # print(fin_ds)
    print(time.time() -t0)
    # break
    # fin_ds.update(fin_ds.sel(ds.coords) + ds)
    
print(time.time() -t0)
# xr.broadcast

fin_ds.isel(time=2).weight.plot()

fin_ds = fin_ds / fin_ds.weight
# fin_ds.isel(time=2).weight.plot()

fin_ds.isel(time=2).gt.plot()

fin_ds.weight

fin_ds

fin_ds[ds.coords] = fin_ds.sel(ds.coords) + ds
xr.addfin_ds + ds
len(lit_mod.outputs)
len(test_patch_coords)
len(test_ds)
test_ds.gt_ds.ds_size



##########################################
self = lit_mod
def iter_item(outputs):
    for chunk in outputs:
        n = chunk['gt'].shape[0]
        for i in range(n):
            yield (
                    chunk['gt'][i],
                    chunk['oi'][i],
                    chunk['preds'][i],
                    chunk['target_obs'][i],
                    chunk['obs_pred'][i],
                    chunk['inp_obs'][i],
            )
    
dses =[ 
        xr.Dataset( {
            'gt': (('time', 'lat', 'lon'), x_gt),
            'oi': (('time', 'lat', 'lon'), x_oi),
            'pred': (('time', 'lat', 'lon'), x_rec),
            'obs_gt': (('time', 'lat', 'lon'), obs_gt),
            'obs_pred': (('time', 'lat', 'lon'), obs_pred),
            'obs_inp': (('time', 'lat', 'lon'), obs_inp),
        }, coords=coords)
    for  (x_gt, x_oi, x_rec, obs_gt, obs_pred, obs_inp), coords
    in zip(iter_item(self.outputs), self.test_patch_coords)
]
import time
t0 = time.time()
fin_ds = reduce(lambda ds1, ds2: xr.merge([ds1, ds2]), [xr.zeros_like(ds[['lat', 'lon', 'time']]) for ds in dses])
print(time.time() - t0)
fin_ds = fin_ds.assign(
    {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
)
for v in dses[0]:
    fin_ds = fin_ds.assign(
        {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
    )


for ds in dses:
    ds_nans = ds.assign(weight=xr.ones_like(ds.gt)).isnull().broadcast_like(fin_ds).fillna(0.)
    xr_weight =xr.DataArray(self.patch_weight, ds.coords, dims=ds.gt.dims) 
    _ds = ds.pipe(lambda dds: dds * xr_weight).assign(weight=xr_weight).broadcast_like(fin_ds).fillna(0.).where(ds_nans==0, np.nan)
    fin_ds = fin_ds + _ds 

ds_minone = dses[-2]

for ds in dses:
    print(str(ds.time.isel(time=2).data))

xr_weight.isel(time=2).plot()
ds.gt.coords
ds.gt.dims
# _ds.sel(time='2013-01-25').gt.plot()
# fin_ds.sel(time='2013-01-25').gt.plot()
fin_ds.sel(time='2013-01-20').gt.plot()
fin_ds.sel(time='2013-01-25').weight.plot()
(fin_ds / fin_ds.weight).sel(time='2013-01-25').pred.plot()
fin_ds.time

fin_ds + _ds 
print("Done")
self.test_xrds = fin_ds
########################################
'''
patch_inference:

total_domain_size
eval_domain_size

patch_size
eval_patch
overlap
stride

eval_patch_size - stride = overlap


data_domain: (220, 220)
    11°x11°
eval_domain: (200, 200)
    10°x10°


patch_size=200
eval_patch_size=180
strides = 40

________________________________________
| 0 |   0      | 0 |     0         |  0|
|___|__________|___|_______________|___|
|   |          |   |               |  0|
| 0 |   1      |1&2|    2          |   |
|___|__________|___|_______________|___|
| 0 |   1&4    |1 2|    2&3        |  0|
|___|__________|4_3|_______________|___|
|   |     4    |   |    3          |  0|
|_0_|__________|___|_______________|___|
| 0 |     0    | 0 |     0         |  0|
|___|__________|___|_______________|___|
 
'''


ds1 = xr.Dataset(
    {'a': (('x',), [1, 2, 3])},
    {'x': (('x',), [1, 2, 3])},
)

ds2 = xr.Dataset(
    {'a': (('x',), [3, 4, 5])},
    {'x': (('x',), [2, 3, 4])},
)

wds = xr.DataArray(
    [1, 2, 1],
    {'x': (('x',), [1, 2, 3])},
    dims=('x',)
)


ds1.weighted(wds)



merge_patch([ds1, ds2])
