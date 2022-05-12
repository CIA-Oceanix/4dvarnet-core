import xarray as xr
import traceback
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import xrft

def get_swath_psd_score(x_t, x, with_fig=False, with_data=False):
    def psd_score(da: xr.DataArray) -> xr.DataArray:
        err = x_t - da
        psd_x_t = (
            x_t.copy()
                .pipe(
                lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='constant', scaling='density'))
                .mean('x_ac')
        ).compute()

        psd_err = (
            err.copy()
                .pipe(
                lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='constant', scaling='density'))
                .mean('x_ac')
        ).compute()
        psd_score = 1 - psd_err / psd_x_t
        return psd_score

    model_score = psd_score(x)

    model_score = (
        model_score.where(model_score.freq_x_al > 0, drop=True).compute()
    )

    psd_plot_data: xr.DataArray = xr.DataArray(
        model_score.data,
        name='PSD score',
        dims=('wl'),
        coords={
            'wl': ('wl', 1 / model_score.freq_x_al.data, {'long_name': 'Wavelength', 'units': 'km'}),
        },
    )
    
    idx = (
            (psd_plot_data.rolling(wl=3, center=True, min_periods=1).mean() > 0.)
            & (psd_plot_data.wl > 10)
    )


    spatial_resolution_model = (
        xr.DataArray(
            # psd_plot_data.sel(var='model').wl,
            psd_plot_data.isel(wl=idx).wl.data,
            dims=['psd'],
            coords={'psd': psd_plot_data.isel(wl=idx).data}
            # coords={'psd': psd_plot_data.sel(var='model').data}
        ).interp(psd=0.5)
    )

    if not with_fig:
        if with_data:
            return spatial_resolution_model, psd_plot_data
        return spatial_resolution_model

    fig, ax = plt.subplots()
    # psd_plot_data.plot.line(x='wl', ax=ax)
    psd_plot_data.rolling(wl=3, center=True, min_periods=1).mean().plot.line('+' ,x='wl', ax=ax)

    # Plot vertical line there
    for i, (sr, var) in enumerate([(spatial_resolution_model, 'model')]):
        plt.axvline(sr, ymin=0, color='0.5', ls=':')
        plt.annotate(f"resolution {var}: {float(sr):.2f} km", (sr * 1.1, 0.1 * i))
        plt.axhline(0.5, xmin=0, color='k', ls='--')
        plt.ylim([0, 1])

    plt.close()
    return fig, spatial_resolution_model

def get_spat_reses(trim_ds, fields=('cal',)):
    spat_reses = []
    chunks = trim_ds.groupby('contiguous_chunk')
    for chunk, g in chunks:
        # print(chunk)

        for c in fields :
            spat_reses.append(
                {
                    'xp_long': c,
                    'spat_res': get_swath_psd_score(g.ssh_model, g[c]).item(),
                    'chunk_nb':g.contiguous_chunk.isel(x_al=0).item()
                }
            )
    spat_res_df = pd.DataFrame(spat_reses)
    return spat_res_df

def make_report():
    try:
        XP_NUM = 14
        grid_metrics = pd.read_csv(f'{XP_NUM}_grid_chain_metrics.csv')
        sw_metrics = pd.read_csv(f'{XP_NUM}_sw_chain_metrics.csv')
        with open(f'{XP_NUM}_figs.pk', 'rb') as f:
            figs = pickle.load(f)
        data = (
                grid_metrics
                .set_index(['xp', 'iter'])
                .drop('Unnamed: 0', axis=1)
                .join(sw_metrics.set_index(['xp', 'iter'])
                .assign(
                    rmse_improvement= lambda df: 1 - df.rmse / df.rmse_pred,
                    grad_rmse_improvement= lambda df: 1 - df.grad_rmse / df.grad_rmse_pred
                )
                .drop('Unnamed: 0', axis=1), rsuffix='_sw')
                .join(pd.DataFrame(figs).set_index(['xp', 'iter']))
                .reset_index().loc[lambda df: df.iter == 0]
                .set_index('xp')
        )

        metrics = [
                'rmse', 'rmse_improvement', 'grad_rmse', 'grad_rmse_improvement',
                'spat_res_mean', 'spat_res_std'
        ]

        grid_xp = lambda df: df.loc[[f'{XP_NUM}_base_duacs', f'{XP_NUM}_direct_obs', f'{XP_NUM}_base_no_sst', f'{XP_NUM}_base_sst']]
        ablation_xp = lambda df: df.loc[[f'{XP_NUM}_base_no_sst', f'{XP_NUM}_no_sst_no_pp', f'{XP_NUM}_no_sst_non_residual']]
        # data.pipe(ablation_xp)[metrics].plot(kind='bar', subplots=True, figsize=(8, 10), layout=(2, 3), legend=False)
        # data.pipe(grid_xp)[metrics].plot(kind='bar', subplots=True, figsize=(8, 10), layout=(2, 3), legend=False)
        with  open('report/violin_all_no_sst.png', 'wb') as f:
            data.loc[f'{XP_NUM}_base_no_sst'].violin_all.savefig(f)

        with  open('report/violin_diff_no_sst.png', 'wb') as f:
            data.loc[f'{XP_NUM}_base_no_sst'].violin_diff.savefig(f)

        with  open('report/grad_ssh_no_sst.png', 'wb') as f:
            data.loc[f'{XP_NUM}_base_no_sst'].ssh.savefig(f)

        with  open('report/err_no_sst.png', 'wb') as f:
            data.loc[f'{XP_NUM}_base_no_sst'].err.savefig(f)


        report = f"""

        ## Results 4dvarnet 5 nadir no sst + CalCNN 
        **Ssh gradients**
        ![](report/grad_ssh_no_sst.png)

        **Residual error after calibration**
        ![](report/err_no_sst.png)

        {data.loc[f'{XP_NUM}_base_no_sst'][metrics].T.to_markdown()}
        
        **Spatial resolution on swath before**
        ![](report/violin_all_no_sst.png)
        ![](report/violin_diff_no_sst.png)

        ## Metrics different grids
        {data.pipe(grid_xp)[metrics].to_markdown()}

        ## Ablation
        {data.pipe(ablation_xp)[metrics].to_markdown()}


        """
        # display(Markdown(report))
        print(report)

    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()


