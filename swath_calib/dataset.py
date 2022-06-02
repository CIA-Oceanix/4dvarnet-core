import torch
import contextlib
import scipy.ndimage as ndi
import xarray as xr
import numpy as np
import pandas as pd

class SmoothSwathDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            swath_data,
            norm_stats=(0., 1.),
            sigmas_obs=(0,*[(i+1)*8 for i in range(10)]),
            sigmas_xb=(0,),
            sigmas_gt=(0,),
            gt_var='ssh_model',
            ref_var='pred',
            xb_var='oi',
        ):
        mean, std = norm_stats
        xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(lambda nda: ndi.gaussian_filter1d(nda, axis=0, sigma=sig, order=0, mode='mirror', truncate=3.0), da)
        pp = lambda da: (da -mean) /std
        swath_data = swath_data.assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
        # swath_data = swath_data.assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time') > 3).cumsum())
        sw_data_w_aug = (
                swath_data
                .groupby('contiguous_chunk')
                .apply(
                    lambda g: g.assign(
                        zeros = lambda _g:  xr.zeros_like(_g.ssh_model)
                    ).assign(
                        err =  lambda _g: _g.syst_error_uncalibrated + _g.wet_tropo_res,
                    ).assign(
                        xb =  lambda _g: _g[xb_var],
                    ).assign(
                        obs = lambda _g:  _g.ssh_model + _g.err
                    ).assign(
                        obs_res = lambda _g: _g.obs - _g.xb,
                        gt_res= lambda ds: ds.ssh_model - ds.xb,
                        ref_res= lambda ds: ds.pred - ds.xb
                    ).assign(
                        **{f'obs_{sig}' : lambda _g, sig=sig: xrgf(pp(_g.obs), sig) for sig in sigmas_obs},
                        **({} if len(sigmas_xb)==0 else {f'xb_{sig}' : lambda _g, sig=sig: xrgf(pp(_g.xb), sig) for sig in sigmas_xb}),
                        **{f'gt_{sig}' : lambda _g, sig=sig: xrgf(pp(_g[gt_var]), sig) for sig in sigmas_gt},
                    )
                )
        )

        sw_res_data = sw_data_w_aug.assign(
                **{
                    f'dobs_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'obs_{sig1}'] - ds[f'obs_{sig2}']
                    for sig1, sig2 in zip(sigmas_obs[:-1], sigmas_obs[1:])
                },
                **({} if len(sigmas_xb)==0 else {
                    f'dxb_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'xb_{sig1}'] - ds[f'xb_{sig2}']
                    for sig1, sig2 in zip(sigmas_xb[:-1], sigmas_xb[1:])
                }),
                **{
                    f'dgt_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'gt_{sig1}'] - ds[f'gt_{sig2}']
                    for sig1, sig2 in zip(sigmas_gt[:-1], sigmas_gt[1:])
                },
        )

        pp_vars = (
                [f'dobs_{sig2}_{sig1}'for sig1, sig2 in zip(sigmas_obs[:-1], sigmas_obs[1:])] 
                + ([f'obs_{sigmas_obs[-1]}'] if len(sigmas_obs)>0 else [])
                + (([f'dxb_{sig2}_{sig1}' for sig1, sig2 in zip(sigmas_xb[:-1], sigmas_xb[1:])]
                    + [f'xb_{sigmas_xb[-1]}']) if len(sigmas_xb)>0 else [])
        )
        gt_vars = (
                [f'dgt_{sig2}_{sig1}'for sig1, sig2 in zip(sigmas_gt[:-1], sigmas_gt[1:])] + [f'gt_{sigmas_gt[-1]}']
        )

        # gt_vars = ['gt_res']
        # ref_var = 'zeros'

        
        # for v in pp_vars:
        #     p(sw_res_data.isel(time=sw_res_data.contiguous_chunk==2)[v])
        all_vars = gt_vars + pp_vars
        mean, std =  norm_stats  if norm_stats is not None else (
                sw_res_data[all_vars].mean(),
                sw_res_data[all_vars].std(),
        )
        # mean, std =train_ds.stats
        # norm_stats=train_ds.stats
        # print(mean)
        # pp_ds = ((sw_res_data[all_vars] - mean) / std).assign(contiguous_chunk=sw_res_data.contiguous_chunk).astype(np.float32)
        pp_ds = sw_res_data[all_vars].assign(contiguous_chunk=sw_res_data.contiguous_chunk).astype(np.float32)

        min_timestep = 300
        self.stats  = mean, std
        self.chunks = list(
                pp_ds.groupby('contiguous_chunk').count()
                .isel(nC=0).pipe(lambda ds: ds.isel(contiguous_chunk=ds[pp_vars[0]] > min_timestep))
                .contiguous_chunk.values
        )

        self.pp_vars = pp_vars 
        self.gt_vars = gt_vars
        self.gt_var = gt_var
        self.ref_var = ref_var
        self.pp_ds = pp_ds
        self.raw_ds = sw_data_w_aug

        self.return_coords = False

    def __len__(self):
        return len(self.chunks)

    @contextlib.contextmanager
    def get_coords(self):
        try:
            self.return_coords = True
            yield
        finally:
            self.return_coords = False

    def __getitem__(self, idx):
        c = self.chunks[idx]
        pp_item_ds = self.pp_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))
        raw_item_ds = self.raw_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))

        if self.return_coords:
            return raw_item_ds
        return (
            pp_item_ds[self.pp_vars].to_array().data,
            pp_item_ds[self.gt_vars].to_array().data,
            raw_item_ds[[self.gt_var]].to_array().data,
            raw_item_ds[[self.ref_var]].to_array().data
        )

