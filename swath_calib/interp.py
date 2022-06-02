import numpy as np
import einops
import xarray as xr
import pandas as pd
import torch
import swath_calib.utils

def torch_interpolate(gv, gt, gx, gy, st, sx, sy):
    """
    interpolate grid values gv on coords (gt, gx, gy)
    to sensor coords (st, sx, sy)
    st is interpolated nearest
    sx, xy bilinearly
    """
    def cp_nearest(s, g0, d):
        return ((s - g0) / d).round().maximum(torch.full_like(s, 0)).long()
    def cp_idx(s, g0, d):
        return ((s - g0) / d).trunc().maximum(torch.full_like(s, 0)).long()

    dt, dx, dy = gt.diff()[0], gx.diff()[0], gy.diff()[0]
    t0, x0, y0 = gt.min(), gx.min(), gy.min()
    nx, ny = len(gx), len(gy)
    ct = cp_nearest(st, t0, dt)
    cx, cy = cp_idx(sx, x0, dx), cp_idx(sy, y0, dy)

    wx = (sx - gx[cx]) / dx
    wy = (sy - gy[cy]) / dy

    cps = gv[
        ct[None, None, ...],
        torch.stack([cx,  (cx+1).minimum(torch.full_like(cx,nx-1))], dim=0)[:, None, ...],
        torch.stack([cy, (cy+1).minimum(torch.full_like(cy,ny-1))], dim=0)[None, :, ...]
    ] 
    _sv = torch.einsum("i...,i... -> ...", torch.stack([1 - wx, wx]) , cps)
    sv = torch.einsum("i...,i... -> ...", torch.stack([1 - wy, wy]) ,_sv)
    return sv


def torch_interpolate_with_fmt(gv, gt, gx, gy, st, sx, sy):
    _st, _sx, _sy = map(torch.flatten, (st, sx, sy))
    inbound = (
        # (_st > gt.min()) & (_st < gt.max()) &
        (_sx > gx.min()) & (_sx < gx.max())
        & (_sy > gy.min()) & (_sy < gy.max())
    )
    msk = (_st.isfinite() & inbound).nonzero(as_tuple=False).squeeze()
    __st, __sx, __sy = map(lambda t: t[msk], (_st, _sx, _sy))
    __v = torch_interpolate(gv, gt, gx, gy, __st, __sx, __sy)
    _v = torch.zeros_like(_st).index_add_(dim=-1, index=msk, source=__v)
    v = _v.reshape_as(st)
    return v



def batch_torch_interpolate(gv, gt, gx, gy, st, sx, sy, bn): 
    def cp_nearest(s, g0, d):
        return ((s - g0) / d).round().maximum(torch.full_like(s, 0)).long()

    def cp_idx(s, g0, d):
        return ((s - g0) / d).trunc().maximum(torch.full_like(s, 0)).long()

    dt, dx, dy = gt.diff()[0,0], gx.diff()[0,0], gy.diff()[0,0]
    t0, x0, y0 = gt.min(1).values[:, None], gx.min(1).values[:, None], gy.min(1).values[:, None]
    nx, ny = len(gx[0,:]), len(gy[0,:])

    ar = torch.arange(len(sx))
    ct = cp_nearest(st, t0, dt)[bn, ar]
    cx, cy = cp_idx(sx, x0, dx)[bn, ar], cp_idx(sy, y0, dy)[bn, ar]

    wx = (sx - gx[bn, cx]) / dx
    wy = (sy - gy[bn, cy]) / dy

    cps = gv[
        bn,
        ct[None, None, ...],
        torch.stack([cx,  (cx+1).minimum(torch.full_like(cx,nx-1))], dim=0)[:, None, ...],
        torch.stack([cy, (cy+1).minimum(torch.full_like(cy,ny-1))], dim=0)[None, :, ...]
    ] 
    _sv = torch.einsum("i...,i... -> ...", torch.stack([1 - wx, wx]) , cps)
    sv = torch.einsum("i...,i... -> ...", torch.stack([1 - wy, wy]) ,_sv)
    return sv

import torch.nn.functional as F


def batch_interp_to_grid(bgv, bgt, bgx, bgy, bsv, bst, bsx, bsy):
    def cp_nearest(s, g0, d):
        return ((s - g0) / d).round().maximum(torch.full_like(s, 0)).long()

    def cp_idx(s, g0, d):
        return ((s - g0) / d).trunc().maximum(torch.full_like(s, 0)).long()

    _sv, _st, _sx, _sy = map(lambda t: torch.flatten(t, start_dim=1), (bsv, bst, bsx, bsy))
    minf = lambda t: einops.reduce(t, 'b ... -> b ()', reduction='min')
    maxf = lambda t: einops.reduce(t, 'b ... -> b ()', reduction='max')
    inbound = (
        # (_st > minf(bgt)) & (_st < maxf(bgt)) &
        (_sx > minf(bgx)) & (_sx < maxf(bgx))
        & (_sy > minf(bgy)) & (_sy < maxf(bgy))
    )

    msk = (_st.isfinite() & inbound).nonzero(as_tuple=True)
    __sv, __st, __sx, __sy = map(lambda t: t[msk], (_sv, _st, _sx, _sy))
    bn, _ = msk
    gt, gx, gy, sv, st, sx, sy, bn = bgt, bgx, bgy, __sv, __st, __sx, __sy, bn

    dt, dx, dy = gt.diff()[0,0], gx.diff()[0,0], gy.diff()[0,0]
    t0, x0, y0 = gt.min(1).values[:, None], gx.min(1).values[:, None], gy.min(1).values[:, None]
    nt, nx, ny = len(gt[0,:]), len(gx[0,:]), len(gy[0,:])

    ar = torch.arange(len(sx))
    ct = cp_nearest(st, t0, dt)[bn, ar]
    # ct = cp_idx(st, t0, dt)[bn, ar]
    cx, cy = cp_idx(sx, x0, dx)[bn, ar], cp_idx(sy, y0, dy)[bn, ar]
    

    fg = torch.zeros_like(bgv).flatten()
    wfg = torch.zeros_like(bgv).flatten()
    fidx= (
            bn * nt * nx * ny 
            + ct *  nx * ny
            + cx  * ny
            + cy 
    )
    fg.index_add_(0, fidx , sv) 
    wfg.index_add_(0, fidx, torch.ones_like(sv)) 
    tgt_fg = fg.where((wfg==0), fg / wfg)
    ggv = tgt_fg.reshape_as(bgv)
    gmsk = wfg.reshape_as(bgv) > 0
    return ggv, gmsk



def batch_torch_interpolate_with_fmt(bgv, bgt, bgx, bgy, bst, bsx, bsy):
    _st, _sx, _sy = map(lambda t: torch.flatten(t, start_dim=1), (bst, bsx, bsy))
    minf = lambda t: einops.reduce(t, 'b ... -> b ()', reduction='min')
    maxf = lambda t: einops.reduce(t, 'b ... -> b ()', reduction='max')
    inbound = (
        # (_st > minf(bgt)) & (_st < maxf(bgt)) &
        (_sx > minf(bgx)) & (_sx < maxf(bgx))
        & (_sy > minf(bgy)) & (_sy < maxf(bgy))
    )

    msk = (_st.isfinite() & inbound).nonzero(as_tuple=True)
    __st, __sx, __sy = map(lambda t: t[msk], (_st, _sx, _sy))
    bn = msk[0]
    sv = batch_torch_interpolate(bgv, bgt, bgx, bgy, __st, __sx, __sy, bn)

    f_msk = (_st.isfinite() & inbound).flatten()
    f_idx = f_msk.nonzero(as_tuple=False).squeeze()
    v = torch.zeros_like(_st.flatten()).index_add_(dim=-1, index=f_idx, source=sv)
    v[~f_msk]= np.nan
    v = v.reshape_as(bst)
    return v

# v = _v.reshape_as(st)

def add_nb_ch(sen_ds):
    min_x_al = np.abs(sen_ds.x_al.diff('time')).min().item()
    ch_nb = np.insert(np.cumsum(sen_ds.x_al.diff('time').pipe(np.abs).data > 1.5 * min_x_al), 0, 0)
    sen_ds = sen_ds.assign( ch_nb=('time', ch_nb.astype(np.float32)))
    tgt_len =  sen_ds.groupby('ch_nb').count().x_al.max().item()
    return sen_ds, tgt_len

def nanmod(nda, m):
    isfin = np.isfinite(nda)
    nda_wo_nan =np.where(isfin, nda, np.zeros_like(nda))
    return np.where(isfin,  nda_wo_nan % m, nda)

def stack_passes(sen_ds, tgt_len, min_len=None):
    passes = []
    if min_len is not None:
        chunks = list(
                sen_ds.groupby('ch_nb').count()
                .isel(nC=0).pipe(lambda ds: ds.isel(ch_nb=ds.to_array().isel(variable=0) > min_len))
                .ch_nb.values
        )
        sen_ds = sen_ds.isel(time=sen_ds.ch_nb.isin(chunks))
        if sen_ds.dims['time']==0:
            return  xr.concat(
                [xr.full_like(
                    sen_ds.reset_index('time').rename({'time': 'idx'}).rename({'time_': 'time'})
                .isel(idx=slice(0,1)), np.nan)], dim='p'
            )
    (
        sen_ds.reset_index('time').rename({'time': 'idx'})
        .rename({'time_': 'time'}).groupby('ch_nb')
        .apply(lambda g: passes.append(
            g.pad(idx=(0, tgt_len - g.dims['idx']), constant_values=np.nan)) or g)
    )
    return xr.concat(passes, dim='p')

def fmt_s_coords(p_ds):
    p_ds = p_ds.broadcast_like(p_ds.ssh_model)
    return tuple(map(torch.from_numpy,(
        p_ds.to_dataframe().pipe(
            lambda df: ((pd.to_datetime(df.time) - pd.to_datetime('2012-10-01')) /pd.to_timedelta(1, 'D'))
        ).pipe(xr.DataArray.from_series).data.astype(np.float32),
        nanmod(p_ds.lat.data.astype(np.float32), 360),
        nanmod(p_ds.lon.data.astype(np.float32), 360),
    )))

def fmt_s_value(p_ds, data_vars, isfinmsk):
    if data_vars is None:
        return None
    p_ds = p_ds.broadcast_like(p_ds.ssh_model)
    _v = p_ds[data_vars].to_array().sum('variable')
    _v = _v.data.astype(np.float32) 
    _v = np.where(isfinmsk, _v, np.nan)
    return torch.from_numpy(_v)

def fmt_g_coords(g_ds):
    g_ds = g_ds.transpose('time', 'lat', 'lon')
    return tuple(map(torch.from_numpy,(
        np.array((pd.to_datetime(g_ds.time.data) - pd.to_datetime('2012-10-01')) /pd.to_timedelta(1, 'D')).astype(np.float32),
        nanmod(g_ds.lat.data.astype(np.float32), 360),
        nanmod(g_ds.lon.data.astype(np.float32), 360),
    )))

def stack(ts):
    tgt_size = torch.stack([torch.tensor(t.shape) for t in ts]).max(0).values.long()
    ps = lambda t: torch.stack(
            [torch.zeros_like(tgt_size), (tgt_size - torch.tensor(t.size())).maximum(torch.zeros_like(tgt_size))]
            , dim=-1).flatten().__reversed__()
    return torch.stack([F.pad(t, [x.item() for x in ps(t)], value=np.nan) for t in ts])


