import numpy as np
import hashlib
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

def full_from_scratch(xp_num, cfgn='base_no_sst', fp="dgx_ifremer"):
    try:
        cfg = utils.get_cfg(f'{cfgn}')

        overrides = [
            '+datamodule.dl_kwargs.shuffle=False',
            f'file_paths={fp}',
            'params.files_cfg.obs_mask_path=${file_paths.new_noisy_swot}',
            'params.files_cfg.obs_mask_var=five_nadirs',
        ]
       

        # Generate grided product on swath
        cfg_4dvar = utils.get_cfg(cfg.fourdvar_cfg, overrides=overrides)

        cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg_4dvar).encode()).hexdigest()
        saved_data_path = Path('tmp') / f'{cfg_hash}.pk'
        print(saved_data_path)
        rms = lambda da: np.sqrt(np.mean(da**2))
        if saved_data_path.exists():
            print('loading  ', str(saved_data_path))
            with open(saved_data_path, 'rb') as f:
                swath_data = pickle.load(f)

        else:
            dm = utils.get_dm(cfg.fourdvar_cfg, add_overrides=overrides)
            model = utils.get_model(cfg.fourdvar_cfg, cfg.fourdvar_mod_ckpt, dm=dm, add_overrides=overrides)

            
            trainer = pl.Trainer(gpus=[7], logger=False)
                
            swath_data = {}
            for stage, dl in [
                    ('test', dm.test_dataloader()),
                    ('val', dm.val_dataloader()),
                    ('train', dm.train_dataloader()),
                ]:
                diag_ds = dl.dataset.datasets[0]
                predictions = trainer.predict(model,dl )
                test_xr_ds = model.build_test_xr_ds([predictions], diag_ds=diag_ds)
                swath_data[stage] = swath_calib.utils.to_swath_data(test_xr_ds).pipe(lambda ds: ds.isel(time=np.isfinite(ds.oi).all('nC')))

            with open(saved_data_path, 'wb') as f:
                pickle.dump(swath_data, f)

        for stage in ['test', 'val', 'train']:
            print(
            f''' \n\n ############################# {cfgn}  {stage} ##########################################
            {rms(swath_data[stage].ssh_model - swath_data[stage].oi).item()=:.2e} m
            {rms(swath_data[stage].ssh_model - swath_data[stage].pred).item()=:.2e} m
            {rms(swath_data[stage].ssh_model - swath_data[stage].gt).item()=:.2e} m
            ''')

        # Train calib model
        train_ds = swath_calib.dataset.SmoothSwathDataset(swath_data['train'], **cfg.swath_ds_cfg) 
        val_ds = swath_calib.dataset.SmoothSwathDataset(swath_data['val'], **cfg.swath_ds_cfg, norm_stats=train_ds.stats) 

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=3)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=3)
        
        net = swath_calib.models.build_net(
                in_channels=len(train_ds.pp_vars),
                out_channels=len(train_ds.gt_vars),
                **cfg.net_cfg
        )
        cal_mod = swath_calib.models.LitDirectCNN(
                net,
                gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
                **cfg.lit_cfg
            )
        logger = pl.loggers.TensorBoardLogger('lightning_logs', name=f'{xp_num}_{cfgn}', version='')
        vcb = swath_calib.versioning_cb.VersioningCallback()
        trainer = pl.Trainer(
            gpus=[7],
            logger=logger,
            callbacks=[
                pl.callbacks.LearningRateMonitor(),
                pl.callbacks.ModelCheckpoint(monitor='val_loss', save_last=True),
                pl.callbacks.StochasticWeightAveraging(),
                pl.callbacks.GradientAccumulationScheduler({1: 4, 10: 8, 15: 16, 20: 32, 30: 64}),
            ],
            log_every_n_steps=10,
            max_epochs=100,
            # max_epochs=2,
        )

        print(pl.utilities.model_summary.summarize(cal_mod, max_depth=3))
        trainer.fit(cal_mod,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl
        )
        uid = str(uuid.uuid4())
        print(uid, " - Best model saved at: ", trainer.checkpoint_callback.best_model_path)
        

        trained_cfg = OmegaConf.merge(
                    OmegaConf.to_container(cfg),
                    {'cal_mod_ckpt':trainer.checkpoint_callback.best_model_path},
                    {'xpnum': xpnum},
                    {'uid':uid},
                    {'cfgn':cfgn},
                    {'xp_num':xp_num},
                    {'src_commit': vcb.setup_hash},
                    {'training_data': str(saved_data_path)},
                
        )

        Path(f'trained_cfgs/{uid}.yaml').write_text(OmegaConf.to_yaml(trained_cfg))

        f_th, sig = 0.01, 1
        ff_net = torch.nn.Sequential(net, swath_calib.models.FourierFilter(f_th, sig))
        print(cal_mod.load_state_dict(torch.load(trained_cfg.cal_mod_ckpt)['state_dict']))
        cal_mod = swath_calib.models.LitDirectCNN(
                ff_net,
                gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
                **cfg.lit_cfg
            )
        # Testing
        fourdvar_dm = utils.get_dm(cfg.fourdvar_cfg, add_overrides=overrides)
        fourdvar_model = utils.get_model(cfg.fourdvar_cfg, cfg.fourdvar_mod_ckpt, dm=fourdvar_dm, add_overrides=overrides)
        grid_metrics = []
        swath_metrics = []
        figs = []
        for niter in range(3):
            trainer.test(fourdvar_model, fourdvar_dm.test_dataloader())[0]
            print(fourdvar_model.latest_metrics)
            grid_metrics = grid_metrics + [{
                    'xp': cfgn,
                    'iter': niter,
                    **fourdvar_model.latest_metrics
            }]
            print(pd.DataFrame(grid_metrics).T.to_markdown())

            # Convert grid estim to swath
            sw_data = swath_calib.utils.to_swath_data(fourdvar_model.test_xr_ds).pipe(lambda ds: ds.isel(time=np.isfinite(ds.pred).all('nC')))

            # Convert grid estim to swath
            cal_ds = swath_calib.dataset.SmoothSwathDataset(sw_data, norm_stats=train_ds.stats, **cfg.swath_ds_cfg) 

            # Estimate on swath using cal model
            cal_data = swath_calib.utils.generate_cal_xrds(cal_ds, cal_mod, trainer)[list(sw_data) + ['cal', 'contiguous_chunk']]

            def sobel(da):
                dx_ac = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 0), da) /2
                dx_al = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 1), da) /2
                return np.hypot(dx_ac, dx_al)
           
            # Grid calibrated swath estimation
            obs_ds = fourdvar_dm.test_ds.datasets[0].obs_mask_ds.ds
            tgt_ds = swath_calib.utils.to_grid(cal_data, obs_ds)

            # Generate new obs on grid with calibrated data
            new_obs = (
                    xr.merge([obs_ds[['five_nadirs']],
                        # obs_ds[['swot_no_noise']],
                        tgt_ds.reindex_like(obs_ds, method='nearest', tolerance=1e-6)
                        ]
                    )
                    .to_array()
                    .mean('variable')
            )

            # TODO: Do inference 
            fourdvar_dm.test_ds.datasets[0].obs_mask_ds.ds = obs_ds.assign(cal=new_obs)
            fourdvar_dm.test_ds.datasets[0].obs_mask_ds.var = 'cal'
        

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
                    [['err', 'pred_err']] 
                    .to_array()
                    .plot.pcolormesh('time', 'x_ac', col='variable', col_wrap=1, figsize=(15, 7))
            ).fig
            plt.close(fig_errs)
            fig_ssh = (
                    cal_data.pipe(add_inter_sw).pipe(lambda d: d.isel(time=d.contiguous_chunk==2))
                    [[ 'ssh_model', 'cal', 'pred']] 
                    .map(lambda da: da.pipe(sobel))
                    # .pipe(sobel)
                    .to_array()
                    .plot.pcolormesh('time', 'x_ac', col='variable', col_wrap=1, figsize=(15, 11))
            ).fig
            plt.close(fig_ssh)
            spat_res_df = swath_calib.report.get_spat_reses(
                cal_data
                .assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
                .assign(
                    syst=lambda d: d.ssh_model + d.syst_error_uncalibrated,
                    tropo=lambda d: d.ssh_model + d.wet_tropo_res,
                    obs=lambda d: d.ssh_model + d.wet_tropo_res + d.syst_error_uncalibrated,
                )
                .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
                .swap_dims(time='x_al', nC='x_ac').drop('time')
            )

            fig_violin_all = sns.violinplot(data=spat_res_df, x='xp_long', y='spat_res').figure
            plt.close(fig_violin_all)
            fig_violin_diff = sns.violinplot(data=spat_res_df.loc[lambda df: df.xp_long.isin(['cal', 'pred'])], x='xp_long', y='spat_res').figure
            plt.close(fig_violin_diff)
            figs.append({
                'xp': cfgn,
                'iter': niter,
                'err': fig_errs,
                'ssh': fig_ssh,
                'violin_all': fig_violin_all,
                'violin_diff': fig_violin_diff,
            })
            print(spat_res_df.groupby('xp_long').spat_res.agg(['mean', 'std']).to_markdown())

            swath_metrics = swath_metrics + [{
                'xp': cfgn,
                'iter': niter,
                'rmse': rms(cal_data.cal - cal_data.ssh_model).item(),
                'rmse_pred': rms(cal_data.pred - cal_data.ssh_model).item(),
                'grad_rmse': cal_data.groupby('contiguous_chunk').apply(lambda g: sobel(g.cal) - sobel(g.ssh_model)).pipe(rms).item(),
                'grad_rmse_pred': cal_data.groupby('contiguous_chunk').apply(lambda g: sobel(g.pred) - sobel(g.ssh_model)).pipe(rms).item(),
            **spat_res_df.loc[lambda df: df.xp_long=='cal'].spat_res.agg({'spat_res_mean': 'mean','spat_res_std': 'std'}).to_dict()
            }]
            
            print(pd.DataFrame(swath_metrics).to_markdown())
        swm_df = pd.DataFrame(swath_metrics)
        swm_df.to_csv(f'tmp/{uid}_sw_chain_metrics.csv')
        gdm_df = pd.DataFrame(grid_metrics)
        gdm_df.to_csv(f'tmp/{uid}_grid_chain_metrics.csv')
        with open(f'tmp/{uid}_figs.pk', 'wb') as f:
            pickle.dump(figs, f)

        
        trained_cfg_w_metrics = OmegaConf.merge(
                OmegaConf.to_container(trained_cfg),
                {'swath_metrics': swm_df.to_dict('record')},
                {'grid_metrics': gdm_df.to_dict('record')},
        )

        Path(f'trained_cfgs/{uid}-w-metrics.yaml').write_text(OmegaConf.to_yaml(trained_cfg_w_metrics))
    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    xp_num=110
    cfgs = swath_calib.configs.register_configs()
    for cfgn in [
        # 'swath_calib_qxp17_aug2_dp240_swot_map_no_sst_ng5x3cas_l2_dp025_00',
        #  'swath_calib_qxp17_aug2_dp240_swot_cal_no_sst_ng5x3cas_l2_dp025_00',
        #  'swath_calib_qxp17_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00',
        #  'swath_calib_qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00',
         'swath_calib_qxp19_aug2_dp240_5nad_map_no_sst_ng5x3cas_l2_dp025_00_dataaug',
         'swath_calib_qxp19_aug2_dp240_5nad_map_sst_ng5x3cas_l2_dp025_00_dataaug'
         'swath_calib_qxp19_aug2_dp240_swot_map_no_sst_ng5x3cas_l2_dp025_00_dataaug',
         'swath_calib_qxp19_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00_dataaug',
         
    ]:
        full_from_scratch(xp_num, cfgn)
