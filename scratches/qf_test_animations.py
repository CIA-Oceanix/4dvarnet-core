
import hydra
import importlib
from hydra.utils import instantiate, get_class, call
import hydra_main
import lit_model_augstate

importlib.reload(lit_model_augstate)
importlib.reload(hydra_main)
from utils import get_cfg, get_dm, get_model
from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
import hydra_config

cfg_n, ckpt = 'full_core', 'results/xpmultigpus/xphack4g_augx4/version_0/checkpoints/modelCalSLAInterpGF-epoch=26-val_loss=1.4156.ckpt'
cfg_n = f"baseline/{cfg_n}"
dm = get_dm(cfg_n, setup=False,
        add_overrides=[
            'file_paths=dgx_ifremer',
        ]


)
mod = get_model(
        cfg_n,
        ckpt,
        dm=dm)
cfg = get_cfg(cfg_n,
        overrides=[
            'file_paths=dgx_ifremer',
        ]
        )
print(OmegaConf.to_yaml(cfg))
lit_mod_cls = get_class(cfg.lit_mod_cls)
runner = hydra_main.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
mod = runner.test(ckpt)
mod.test_figs['maps']
import cartopy.crs as ccrs
import matplotlib.animation as mpa

fig = mod.test_xr_ds.isel(time=0).to_array().plot.pcolormesh(
        x='lon', y='lat', col='variable', col_wrap=2)
newfig = mod.test_xr_ds.isel(time=1).to_array().plot.pcolormesh(
        x='lon', y='lat', col='variable', col_wrap=2)


import holoviews as hv
from holoviews import opts
hv.extension('matplotlib')
hvds = hv.Dataset(mod.test_xr_ds)
images = hv.Layout([
        hvds
        .to(hv.QuadMesh, ['lon', 'lat'], v).relabel(v)
        .options(cmap='RdYlBu')
        for v in ['pred', 'gt', 'oi', 'obs_inp']
        ]).cols(2).opts(sublabel_format="")
hv.output(images, holomap='mp4', fps=3)


plt.cm.get_cmap("RdYlBu")

def get_grid_fig(to_plot_ds):
clims = Clim(to_plot_ds)
hv_layout = hv.Layout([
    hv.Dataset(
        to_plot_ds, ['lon', 'lat'], var
    ).to(
        hv.QuadMesh, kdims=['lon', 'lat']
    ).relabel(
        f'{var}'
    ).options(
        colorbar=True,
        cmap='PiYG',
        clim=clims[var],
        aspect=2
    )
    for var in to_plot_ds
]).cols(2)
