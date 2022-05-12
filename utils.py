import xarray as xr
import hydra
from hydra.utils import instantiate, get_class, call
from pathlib import Path
import hydra_main

def coords_to_dim(ds, dims=('time',), drop=('x',)):
    df = ds.to_dataframe()

    for d in dims:
        df = df.set_index(d, append=True)
    return (
        df.reset_index(level=drop, drop=True)
            .pipe(lambda ddf: xr.Dataset.from_dataframe(ddf))
    )

def reindex(ds, dims=('time', 'lat', 'lon')):
    df = ds.to_dataframe().reset_index()

    for i, d in enumerate(dims):
        df = df.set_index(d, append=i>0)
    return df.pipe(lambda ddf: xr.Dataset.from_dataframe(ddf))


def get_cfg(xp_cfg, overrides=None):
    overrides = overrides if overrides is not None else []
    def get():
        cfg = hydra.compose(config_name='main', overrides=
            [
                f'xp={xp_cfg}',
                'file_paths=jz',
                'entrypoint=train',
            ] + overrides
        )

        return cfg
    try:
        with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
            return get()
    except ValueError as e:
        return get()

def get_model(xp_cfg, ckpt, dm=None, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides =  overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    if dm is None:
        dm = instantiate(cfg.datamodule)
    runner = hydra_main.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner._get_model(ckpt)
    return mod

def get_dm(xp_cfg, setup=True, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides = overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    dm = instantiate(cfg.datamodule)
    if setup:
        dm.setup()
    return dm
