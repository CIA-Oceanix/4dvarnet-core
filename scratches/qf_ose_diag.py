import utils
import traceback


base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    'file_paths={fp}'
]



def run1():
    try:
        import sys
        sys.path.append('../4dvarnet-starter')
        import src
        import inspect
        import importlib
        import src.models
        importlib.reload(src.models)
        import hydra
        from pathlib import Path
        xpdir = '/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2022-11-23/15-23-51'

        with hydra.initialize(config_path='..' /Path(xpdir).relative_to(Path('..').resolve().absolute()) /'.hydra', version_base="1.2"):
            cfg = hydra.compose('config.yaml', overrides=['trainer.logger=False'])

        trainer = hydra.utils.call(cfg.trainer)()
        lit_mod = hydra.utils.call(cfg.model)()
        # print(lit_mod.solver.solver_step)
        # print(inspect.getsource(lit_mod.solver.solver_step))
        ckpt = '/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2022-11-22/17-28-14/base/checkpoints/best.ckpt'
        ckpt = xpdir +'/base/checkpoints/best.ckpt'
        lit_mod.load_state_dict(torch.load(ckpt)['state_dict'])
                
        dm = hydra.utils.call(cfg.datamodule)()
        dm.setup('test')

        trainer.test(lit_mod, datamodule=dm)
        # trainer.test(lit_mod, dataloaders=dm.val_dataloader())
        m, s = dm.norm_stats
        # lit_mod.test_data.isel(time=0).ssh.plot()
        # lit_mod.test_data.isel(time=0).rec_ssh.plot()
        print(
                lit_mod
                .test_data.isel(time=slice(7, -7))
                .pipe(lambda ds: (ds.rec_ssh -ds.ssh)*s).pipe(lambda da: da**2).mean().pipe(np.sqrt)
        )
        tdat = lit_mod.test_data.isel(time=slice(5, -5)) *s +m
        dm.test_ds.da.sel(variable='tgt').pipe(lambda da: da.sel(tdat.coords) - tdat.ssh).isel(time=1).plot()
        psdda, lx, lt = metrics.psd_based_scores(tdat.rec_ssh, tdat.ssh,)
        print(lx, lt)
        errt, errmap, mu, sig = metrics.rmse_based_scores(tdat.rec_ssh, tdat.ssh,)
        print(mu, sig)
        cfg = utils.get_cfg(base_cfg)
        dm = utils.get_dm(base_cfg, add_overrides=overrides)
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        fn = run1

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    locals().update(main())
