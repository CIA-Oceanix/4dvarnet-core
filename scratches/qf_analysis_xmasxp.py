import xarray as xr
from scipy import ndimage
import pandas as pd
import numpy as np

path_oi_4nadir = '../sla-data-registry/NATL60/NATL/oi/ssh_NATL60_4nadir.nc'
path_oi_swot_4nadir = '../sla-data-registry/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc'
path_ref_inst = '../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc'
path_ref_daily = '../sla-data-registry/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc'

ois = """
five_nadirs_oi.nc
four_nadirs_oi.nc
swot_nadirs_new_errors_no_wet_tropo_oi.nc
swot_nadirs_new_errors_w_wet_tropo_oi.nc
swot_nadirs_no_noise_oi.nc
swot_nadirs_old_errors_oi.nc
""".strip().splitlines()

trained_model = """
xpxmas_test/4nad_cal_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/4nad_cal_sst_inc_ngrad/version_0/test.nc
xpxmas_test/4nad_cal_sst_inc_ngrad/version_1/test.nc
xpxmas_test/4nad_map_no_sst/version_0/test.nc
xpxmas_test/4nad_map_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/4nad_map_sst_inc_ngrad/version_0/test.nc
xpxmas_test/5nad_cal_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/5nad_cal_sst_inc_ngrad/version_0/test.nc
xpxmas_test/5nad_cal_sst_inc_ngrad/version_1/test.nc
xpxmas_test/5nad_map_no_sst/version_0/test.nc
xpxmas_test/5nad_map_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/5nad_map_sst_inc_ngrad/version_0/test.nc
xpxmas_test/errs_cal_no_sst/version_0/test.nc
xpxmas_test/errs_cal_no_sst_dec_lr/version_0/test.nc
xpxmas_test/errs_cal_no_sst_dec_lr/version_1/test.nc
xpxmas_test/errs_cal_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/errs_cal_sst/version_0/test.nc
xpxmas_test/errs_cal_sst_dec_lr/version_0/test.nc
xpxmas_test/errs_cal_sst_inc_ngrad/version_0/test.nc
xpxmas_test/errs_cal_sst_inc_ngrad/version_1/test.nc
xpxmas_test/errs_map_no_sst/version_0/test.nc
xpxmas_test/errs_map_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/errs_map_no_sst_inc_ngrad/version_1/test.nc
xpxmas_test/errs_map_sst/version_0/test.nc
xpxmas_test/errs_map_sst/version_1/test.nc
xpxmas_test/errs_map_sst_dec_lr/version_0/test.nc
xpxmas_test/errs_map_sst_dec_lr/version_1/test.nc
xpxmas_test/errs_map_sst_inc_ngrad/version_0/test.nc
xpxmas_test/no_err_cal_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/no_err_cal_sst_inc_ngrad/version_0/test.nc
xpxmas_test/no_err_map_no_sst/version_0/test.nc
xpxmas_test/no_err_map_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/no_err_map_sst_inc_ngrad/version_0/test.nc
xpxmas_test/old_errs_cal_no_sst/version_0/test.nc
xpxmas_test/old_errs_cal_no_sst/version_1/test.nc
xpxmas_test/old_errs_cal_no_sst_dec_lr/version_0/test.nc
xpxmas_test/old_errs_cal_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/old_errs_cal_sst/version_0/test.nc
xpxmas_test/old_errs_cal_sst_dec_lr/version_0/test.nc
xpxmas_test/old_errs_cal_sst_inc_ngrad/version_0/test.nc
xpxmas_test/old_errs_map_no_sst/version_0/test.nc
xpxmas_test/old_errs_map_no_sst_dec_lr/version_0/test.nc
xpxmas_test/old_errs_map_no_sst_inc_ngrad/version_0/test.nc
xpxmas_test/old_errs_map_sst/version_0/test.nc
xpxmas_test/old_errs_map_sst_dec_lr/version_0/test.nc
xpxmas_test/old_errs_map_sst_dec_lr/version_1/test.nc
xpxmas_test/old_errs_map_sst_inc_ngrad/version_0/test.nc
""".strip().splitlines()

'xpxmas_test/5nad_map_no_sst/version_0/test.nc',
'xpxmas_test/errs_cal_no_sst/version_0/test.nc',

'xpxmas_test/5nad_map_sst_inc_ngrad/version_0/test.nc'
'xpxmas_test/errs_map_sst/version_1/test.nc',

def main():

    # src data
    oi_4nad_ds = xr.open_dataset(path_oi_4nadir)
    oi_swot_ds = xr.open_dataset(path_oi_swot_4nadir)


    def extract_feat():
        import re

        v_ex = '.+version_(\d+).+'
        cal_ex = '.+(cal|map).+'
        obs_ex = '.+(old_errs|no_err|(?<=/)errs|4nad|5nad).+'
        sst_ex = '.+(no_sst|(?<!no_)sst).+'
        train_ex = '.+(inc_ngrad|dec_lr).+'
        
        feats = {}        
        for td in trained_model:
            td_feats = {}
            m = re.match(v_ex, td)
            if m is not None:
                td_feats['v'] = m.group(1)
            m = re.match(cal_ex, td)
            if m is not None:
                td_feats['cal'] = m.group(1)
            m = re.match(train_ex, td)
            if m is not None:
                td_feats['train'] = m.group(1)
            m = re.match(obs_ex, td)
            if m is not None:
                td_feats['obs'] = m.group(1)
            m = re.match(sst_ex, td)
            if m is not None:
                td_feats['sst'] = m.group(1)

            
            print(td, td_feats)

            feats[td] = td_feats

        return feats
    ## oi perf 
    ### load first mod
    ### load all ois
    ### compute all metrics

    oi_ds =  (
        xr.open_dataset(trained_model[0])[['gt']].isel(time=slice(None, -1))
        .pipe(lambda ds: ds.assign(
            duacs_4nad=(list(ds.dims.keys()), oi_4nad_ds.interp(**{c: cd['data']  for c, cd in ds.to_dict()['coords'].items()}).ssh_mod.values),
            duacs_swot=(list(ds.dims.keys()), oi_swot_ds.interp(**{c: cd['data']  for c, cd in ds.to_dict()['coords'].items()}).ssh_mod.values),
        ))
    )

    def sobel_grid(da):
        dlat = da.pipe(lambda da:  da.groupby('time').apply(lambda da: ndimage.sobel(da, da.dims.index('lat')))) / 5
        dlon = da.pipe(lambda da:  da.groupby('time').apply(lambda da: ndimage.sobel(da, da.dims.index('lon')))) / 5
        return np.hypot(dlat, dlon)

    for oi in ois:
        cust_oi = xr.open_dataset(oi)

        oi_ds = oi_ds.assign({
            oi.split('.')[0]:(list(oi_ds.dims.keys()), cust_oi.interp(**{c: cd['data']  for c, cd in oi_ds.to_dict()['coords'].items()}).ssh.values)
        })


    ## td perf
    rmse  = {}
    rmse_grad  = {}
    rmse_swath  = {}
    rmse_grad_swath  = {}
    for td in trained_model:
        td_ds = xr.open_dataset(td)

        rmse[td] = np.sqrt((td_ds.pred - td_ds.gt).pipe(lambda ds: ds**2).mean().item())
        rmse_grad[td] = np.sqrt((sobel_grid(td_ds.pred) - sobel_grid(td_ds.gt)).pipe(lambda ds: ds**2).mean().item())

        swath_approx = td_ds.obs_pred if 'cal' in td else td_ds.pred
        rmse_swath[td] = np.sqrt(np.mean((swath_approx.values[np.isfinite(td_ds.obs_gt)] - td_ds.obs_gt.values[np.isfinite(td_ds.obs_gt)])**2))

        gt_grad = sobel_grid(td_ds.obs_gt)
        rmse_grad_swath[td] = np.sqrt(np.mean((sobel_grid(swath_approx).values[np.isfinite(gt_grad)] - gt_grad.values[np.isfinite(gt_grad)])**2))


    rmse_td_df = pd.DataFrame([rmse]).T.rename({0: 'rmse'}, axis=1)
    rmse_grad_td_df = pd.DataFrame([rmse_grad]).T.rename({0: 'rmse_grad'}, axis=1)
    rmse_swath_td_df = pd.DataFrame([rmse_swath]).T.rename({0: 'rmse_swath'}, axis=1)
    rmse_grad_swath_td_df = pd.DataFrame([rmse_grad_swath]).T.rename({0: 'rmse_grad_swath'}, axis=1)

    feats = extract_feat()
    td_df = pd.concat([
        rmse_td_df,
        rmse_swath_td_df,
        rmse_grad_td_df,
        rmse_grad_swath_td_df,
    ], axis=1).join(pd.DataFrame(feats).T).reset_index(drop=True)

    td_df.to_csv('xmas_xp.csv')
    rmse_ois_df = (oi_ds.drop('gt') - oi_ds.gt).pipe(lambda ds: ds**2).to_dataframe().mean().pipe(lambda df: np.sqrt(df))

    pd.DataFrame(feats).T.columns
    pd.concat([rmse_td_df, rmse_ois_df])
    return locals()


def main2():
    no_errs = next((k for k,v in feats.items() if v['obs']=='no_err' and v['cal']=='map' and v['sst']=='no_sst'  and 'train' not in  v))
    no_errs_sst = next((k for k,v in feats.items() if v['obs']=='no_err' and v['cal']=='map' and v['sst']=='sst' and v.get('train')=='inc_ngrad'))
    fivenad = next((k for k,v in feats.items() if v['obs']=='5nad' and v['cal']=='map' and v['sst']=='sst'  and v.get('train')=='inc_ngrad'))
    fivenad_sst = next((k for k,v in feats.items() if v['obs']=='5nad' and v['cal']=='map' and v['sst']=='no_sst'  and 'train' not in  v))
    old_err_cal = next((k for k,v in feats.items() if v['obs']=='old_errs' and v['cal']=='map' and v['sst']=='no_sst'  and 'train' not in  v))
    old_err_map = next((k for k,v in feats.items() if v['obs']=='old_errs' and v['cal']=='cal' and v['sst']=='no_sst'  and 'train' not in  v))
    old_err_map_sst = next((k for k,v in feats.items() if v['obs']=='old_errs' and v['cal']=='map' and v['sst']=='sst'  and v.get('train')=='dec_lr'))
    errs_cal = next((k for k,v in feats.items() if v['obs']=='errs' and v['cal']=='cal' and v['sst']=='no_sst'  and 'train' not in  v))
    errs_map = next((k for k,v in feats.items() if v['obs']=='errs' and v['cal']=='map' and v['sst']=='no_sst'  and v.get('train')=='inc_ngrad' and v['v']=='1'))
    errs_map_sst = next((k for k,v in feats.items() if v['obs']=='errs' and v['cal']=='map' and v['sst']=='sst'  and v.get('train')=='inc_ngrad'))

    import hvplot.xarray
    plot_kwargs = dict(clim=(-0.5, 0.5))
    ds_noerr = xr.open_dataset(no_errs).assign(err = lambda ds: ds.gt - ds.pred)
    ds_noerr.err.isel(time=4).plot(**plot_kwargs)

    ds_fivenad = xr.open_dataset(fivenad).assign(err = lambda ds: ds.gt - ds.pred)
    ds_fivenad.err.isel(time=4).plot()

    ds_errs = xr.open_dataset(errs_map).assign(err = lambda ds: ds.gt - ds.pred)
    ds_errs.err.isel(time=4).plot()



    (
            ds_noerr.err.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + ds_fivenad.err.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + ds_errs.err.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
    )

    (
            ds_noerr.obs_inp.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + ds_fivenad.obs_inp.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + ds_errs.obs_inp.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
    )

    (
            (ds_noerr - ds_fivenad).err.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + (ds_fivenad - ds_errs).err.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + (ds_noerr - ds_errs).err.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
    )
    (
            ds_noerr.pred.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + ds_fivenad.pred.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + ds_errs.pred.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
    )
    (
            sobel_grid(ds_noerr.gt).isel(time=4).hvplot.image(width=450, cmap='RdBu') \
            + sobel_grid(ds_noerr.pred).isel(time=4).hvplot.image(width=450, cmap='RdBu') \
            + sobel_grid(ds_fivenad.pred).isel(time=4).hvplot.image(width=450, cmap='RdBu') \
            + sobel_grid(ds_errs.pred).isel(time=4).hvplot.image(width=450, cmap='RdBu') \
    )
    (
            sobel_grid(ds_noerr.obs_gt).isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            # + sobel_grid(ds_fivenad.obs_inp).isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            # + sobel_grid(ds_errs.obs_inp).isel(time=4).hvplot.image(width=500, cmap='RdBu') \
    )
    (ds_noerr.obs_inp.isel(time=4).hvplot.image(width=500, cmap='RdBu') \
            + ds_errs.obs_inp.isel(time=4).hvplot.image(width=500, cmap='RdBu'))

    ds = xr.open_dataset(old_err_cal).assign(err = lambda ds: ds.gt - ds.pred)
    ds.assign(err = lambda ds: ds.gt - ds.pred).err.isel(time=4).plot()
    ds.assign(err = lambda ds: ds.gt - ds.pred).err.isel(time=4).hvplot()
    ds = xr.open_dataset(old_err_map)
    ds.assign(err = lambda ds: ds.gt - ds.pred).err.isel(time=4).plot()
    ds = xr.open_dataset(errs_cal)
    ds.assign(err = lambda ds: ds.gt - ds.pred).err.isel(time=4).plot()
    ds.obs_gt.isel(time=4).plot()
