import numpy as np
import scipy.signal
import kornia
import xrft
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndi
from omegaconf import OmegaConf
import xarray as xr
import pandas as pd
import torch
import pickle
import swath_calib.configs
import swath_calib.utils
import swath_calib.models
import swath_calib.dataset
import swath_calib.versioning_cb
import swath_calib.report
import pytorch_lightning as pl
from pathlib import Path
import time
import utils
import uuid


import importlib
importlib.reload(swath_calib.configs)
importlib.reload(utils)
importlib.reload(swath_calib.utils)

rms = lambda da: np.sqrt(np.mean(da**2))


trained_cfgs = list(Path('trained_cfgs').glob('*metrics.yaml'))

uid = '01b8b31e-ef6e-4ebe-bc59-b62a2c253451'
# uid = '8e039ce7-5dc9-46d9-ac60-df4abcabe86b'
for c in trained_cfgs:
    c = OmegaConf.load(c)
    print(OmegaConf.masked_copy(c, ['uid','cal_mod_ckpt', 'fourdvar_cfg', 'training_data'] ))
    if c.uid == uid:
        break

cfg =c
overrides = [
    '+datamodule.dl_kwargs.shuffle=False',
    f'file_paths=dgx_ifremer',
    'params.files_cfg.obs_mask_path=${file_paths.new_noisy_swot}',
    'params.files_cfg.obs_mask_var=five_nadirs',
]
trainer = pl.Trainer(gpus=[7])
fourdvar_dm = utils.get_dm(cfg.fourdvar_cfg, add_overrides=overrides)
fourdvar_model = utils.get_model(cfg.fourdvar_cfg, cfg.fourdvar_mod_ckpt, dm=fourdvar_dm, add_overrides=overrides)
trainer.test(fourdvar_model, fourdvar_dm.test_dataloader())[0]

OmegaConf.create(pd.DataFrame([fourdvar_model.latest_metrics]).to_dict('record'))
with open(c.training_data, 'rb') as f:
    swath_data = pickle.load(f)

train_ds = swath_calib.dataset.SmoothSwathDataset(swath_data['train'], **c.swath_ds_cfg) 
val_ds = swath_calib.dataset.SmoothSwathDataset(swath_data['val'], **c.swath_ds_cfg, norm_stats=train_ds.stats) 
net = swath_calib.models.build_net(
        in_channels=len(train_ds.pp_vars),
        out_channels=len(train_ds.gt_vars),
        **c.net_cfg
)
_cal_mod = swath_calib.models.LitDirectCNN(
        net,
        gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
        **c.lit_cfg
)
print(_cal_mod.load_state_dict(torch.load(c.cal_mod_ckpt)['state_dict']))
test_ds = swath_calib.dataset.SmoothSwathDataset(swath_data['test'], **c.swath_ds_cfg, norm_stats=train_ds.stats) 


ff_net = torch.nn.Sequential(net, FourierFilter(f_th, sig))
cal_mod = swath_calib.models.LitDirectCNN(
        ff_net,
        gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
        **c.lit_cfg
)
trainer = pl.Trainer(gpus=[7])
_cal_data = swath_calib.utils.generate_cal_xrds(test_ds, _cal_mod, trainer)[list(swath_data['test']) + ['cal', 'contiguous_chunk']]
cal_data = swath_calib.utils.generate_cal_xrds(test_ds, cal_mod, trainer)[list(swath_data['test']) + ['cal', 'contiguous_chunk']]

fig1 = f(_cal_data)
plt.close(fig1)
fig2 = f(cal_data)
plt.close(fig2)

def f(cal_data=cal_data):
    swath_metrics = [{
        'rmse': rms(cal_data.cal - cal_data.ssh_model).item(),
        'grad_rmse': cal_data.groupby('contiguous_chunk').apply(lambda g: sobel(g.cal) - sobel(g.ssh_model)).pipe(rms).item(),
    }]
    print(swath_metrics)

    fourier_filter = fourier_filter2
    fft_data = cal_data.groupby('contiguous_chunk').apply(
            lambda g: g.assign(ff_cal= fourier_filter(f_th, sig)(g.cal)))

    spat_res_df = swath_calib.report.get_spat_reses(
        fft_data
        .assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
        .assign(
            syst=lambda d: d.ssh_model + d.syst_error_uncalibrated,
            tropo=lambda d: d.ssh_model + d.wet_tropo_res,
            obs=lambda d: d.ssh_model + d.wet_tropo_res + d.syst_error_uncalibrated,
        )
        .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
        .swap_dims(time='x_al', nC='x_ac').drop('time')
        , ['ff_cal', 'cal', 'pred']
    )
    print(spat_res_df.groupby('xp_long').spat_res.agg(['mean', 'std']).to_markdown())

    swath_metrics = [{
        'rmse': rms(fft_data.cal - fft_data.ssh_model).item(),
        'grad_rmse': fft_data.groupby('contiguous_chunk').apply(lambda g: sobel(g.cal) - sobel(g.ssh_model)).pipe(rms).item(),
        'ff_rmse': rms(fft_data.ff_cal - fft_data.ssh_model).item(),
        'ff_grad_rmse': fft_data.groupby('contiguous_chunk').apply(lambda g: sobel(g.ff_cal) - sobel(g.ssh_model)).pipe(rms).item(),
    }]
    print(swath_metrics)


    fig_violin_all = sns.violinplot(data=spat_res_df, x='xp_long', y='spat_res').figure
    return fig_violin_all


def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 0), da) /2
    dx_al = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 1), da) /2
    return np.hypot(dx_ac, dx_al)

add_inter_sw = lambda ds:(
            ds
        .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
        .swap_dims(nC='x_ac')
        .reindex(x_ac=np.arange(-60, 62, 2), fill_value=np.nan)
)
v = 'cal'
chunk=2
fig_errs = (
        cal_data.pipe(add_inter_sw).pipe(lambda d: d.isel(time=d.contiguous_chunk==chunk))
        .assign(err=lambda d: d[v] - d.ssh_model)
        .assign(pred_err=lambda d: d.pred - d.ssh_model)
        [[
            'err',
            'pred_err'
        ]] 
        .to_array()
        .plot.pcolormesh('time', 'x_ac', col='variable', col_wrap=1, figsize=(15, 7))
).fig

plt.close(fig_errs)
fig_ssh = (
        cal_data.pipe(lambda d: d.isel(time=d.contiguous_chunk==2))
        .pipe(add_inter_sw)
        [[ 'ssh_model', 'cal', 'pred']] 
        .map(lambda da: da.pipe(sobel))
        # .pipe(sobel)
        .to_array()
        .plot.pcolormesh('time', 'x_ac', col='variable', col_wrap=1, figsize=(15, 11))
).fig
plt.close(fig_ssh)


with torch.no_grad():
    batch = cal_mod.transfer_batch_to_device(next(iter(test_dl)), cal_mod.device, 1)
    out= cal_mod(batch)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=3)

def fourier_filter1(f_th, sig):
    def _f(da):
        data = (
                xrft.fft(da.swap_dims(time='x_al').drop('time'), dim='x_al')
                # .isel(x_ac=0)
                .pipe(lambda x: x.where(x.freq_x_al < f_th, ndi.gaussian_filter1d(x, sigma=sig, axis=0)))
                # .pipe(lambda x: x.where(x.freq_x_al < f_th, ndi.median_filter(x, size=10)))
                .pipe(lambda da:xrft.ifft( da, dim='freq_x_al'))
                .pipe(np.real)
                .data
        )
        return  xr.DataArray(data, dims=da.dims, coords=da.coords)
    return _f


da = cal_data.isel(time=cal_data.contiguous_chunk==2).cal

def fourier_filter2(f_th, sig):
    def _f(da):
        data = (
                xrft.fft(da.swap_dims(time='x_al').drop('time'), dim='x_al')
                # .isel(x_ac=0)
                .pipe(lambda x: (
                    xrft.ifft(x.where(x.freq_x_al < f_th, xr.zeros_like(x)), dim='freq_x_al')
                    + ndi.gaussian_filter1d(
                        xrft.ifft(x.where(x.freq_x_al > f_th, 0.*x), dim='freq_x_al')
                        , sigma=sig, axis=0)
                    )
                )

                # .pipe(lambda x: x.where(x.freq_x_al < f_th, ndi.median_filter(x, size=10)))
                .pipe(np.real)
                .data
        )
        return  xr.DataArray(data, dims=da.dims, coords=da.coords)
    return _f

