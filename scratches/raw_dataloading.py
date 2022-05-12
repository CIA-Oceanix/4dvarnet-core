# %% Imports
import traceback
import os
from dataclasses import dataclass, field
from functools import reduce, lru_cache
from pathlib import Path
from typing import Any

import holoviews as hv
import hydra
import learn2learn as l2l
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import shortuuid
import torch
import torch.nn.functional as F
import xarray as xr
from holoviews.plotting import mpl
# %%
from einops import repeat
from hydra.core.config_store import ConfigStore
from hydra.experimental import initialize, compose
from hydra.types import TargetConf
from hydra.utils import instantiate, call
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from src.commons.coords_to_dim import coords_to_dim


@dataclass
class SlicerCfg:
    path: str
    _target_: str


@dataclass
class DataCfg:
    srcs: dict = field(default_factory=lambda: {
        'swot': SlicerCfg(path='${adp:../../data/zarr/swot}', _target_='src.data_processing.get_slice.get_swot_slice'),
        'nadirs': {name: SlicerCfg(path='${adp:../../data/zarr/nadir/' + name + '}',
                                   _target_='src.data_processing.get_slice.get_nadir_slice')
                   for name in ['swot', 'j1', 'g2', 'en', 'tpn']},
        # 'oi': SlicerCfg(path='${adp:../../data/raw/DUACS-OI_maps/ssh_model/ssh_sla_boost_NATL60_en_j1_tpn_g2.nc}',
        #                 _target_='src.data_processing.get_slice.get_swot_slice'),
        # 'natl60': SlicerCfg(path='${adp:../../data/raw/NATL60_regular_grid/1_10/natl60CH_H.nc}',
        #                     _target_='src.data_processing.get_slice.get_natl_slice'),
    })
    specs: dict = field(
        default_factory=lambda: {
            'swot': {
                'fields': ('ssh_model', 'roll_err', 'x_al', 'x_ac', 'lat', 'lon', 'time'),
                'dims': {'default': ('time', 'nC'), 'x_al': ('time',), 'time': ('time',)},
                'custom_fields': {'uncal': ['ssh_model', 'roll_err']}},
            'nadirs': {
                'fields': ('ssh_model', 'lat', 'lon', 'time'),
                'dims': {'default': ('time',)}
            },
            # 'oi': {
            #     'dims': {'default': ('time', 'y', 'x'), 'lat': ('y',), 'lon': ('x',), 'time': ('time',)},
            #     'fields': ('ssh', 'lat', 'lon', 'time')},
            # 'natl60': {
            #     'dims': {'default': ('time', 'y', 'x'), 'lat': ('y',), 'lon': ('x',), 'time': ('time',)},
            #     'fields': ('H', 'lat', 'lon', 'time')},

        })

    preprocess: dict = field(
        default_factory=lambda: {
            'swot': {
                'offset': {
                    'time': 1.3490496e+18
                },
                'scale': {
                    'x_al': 1000.  ,
                    'x_ac': 30.,
                    'time': 10 ** 9 * 3600 * 6,
                    'lat': 4.,
                    'lon': 4.,
                }
            },
            'nadirs': {
                'offset': {
                    'time': 1.3490496e+18
                },
                'scale': {
                    'time': 10 ** 9 * 3600 * 6,
                    'lat': 4.,
                    'lon': 4.,
                }
            },
        })


@dataclass
class DatasetCfg:
    time_min: str = "2012-10-01"
    time_max: str = "2013-09-30"
    time_periods: int = 365
    time_window: int = 10

    lat_min: int = 30
    lat_max: int = 40
    lat_periods: int = 1
    lat_window: int = 1

    lon_min: int = 295
    lon_max: int = 305
    lon_periods: int = 1
    lon_window: int = 1




@dataclass
class TrainingCfg:
    train_ds_cfg: list = field(default_factory=lambda: [
        DatasetCfg(time_max='2012-11-21', time_periods=50), DatasetCfg(time_min='2013-01-29', time_periods=245)
    ])
    val_ds_cfg: list = field(default_factory=lambda: [
        DatasetCfg(time_min='2012-11-30', time_max='2012-12-20', time_periods=20)
    ])
    test_ds_cfg: list = field(default_factory=lambda: [
        DatasetCfg(time_min='2012-12-30', time_max='2013-01-19', time_periods=20)
    ])




@dataclass
class Config:
    data: DataCfg = DataCfg()
    training: TrainingCfg = TrainingCfg()






# %%
class XrDataset(Dataset):
    # CACHE = {}
    def __init__(self, data_cfg: DataCfg, ds_cfg: DatasetCfg):
        self.ds_cfg = ds_cfg
        self.data_cfg = data_cfg
        self.ds_size = (
            (self.ds_cfg.time_periods - self.ds_cfg.time_window),
            (self.ds_cfg.lat_periods - self.ds_cfg.lat_window + 1),
            (self.ds_cfg.lon_periods - self.ds_cfg.lon_window + 1),
        )
        self.times = pd.date_range(self.ds_cfg.time_min, self.ds_cfg.time_max, periods=self.ds_cfg.time_periods)
        self.lats = np.linspace(self.ds_cfg.lat_min, self.ds_cfg.lat_max, self.ds_cfg.lat_periods + 1)
        self.lons = np.linspace(self.ds_cfg.lon_min, self.ds_cfg.lon_max, self.ds_cfg.lon_periods + 1)

    def __len__(self):
        return self.ds_size[0] * self.ds_size[1] * self.ds_size[2]

    @staticmethod
    def fetch_fields(slice_args, src_cfg, field_specs):
        # print(src_cfg, slice_args)
        slice_ds = call(src_cfg, **slice_args)[list(field_specs.fields)].compute()
        fields = {}
        mask = None
        for fn in field_specs.fields:
            # print(f'fetching field {fn} with {field_specs}')
            tgt_dims = field_specs.dims['default']
            data = repeat(
                slice_ds[fn].data,
                ' '.join(field_specs.dims.get(fn, tgt_dims)) + '->' + '(' + ' '.join(tgt_dims) + ')',
                **{dk: dv for dk, dv in slice_ds.dims.items() if dk in tgt_dims}
            )
            if mask is None:
                mask = ~np.isnan(data)
            else:
                mask = mask & ~np.isnan(data)
            fields[fn] = data.astype(np.float32)

        for cfn in field_specs.get("custom_fields", {}):
            fields[cfn] = np.add(*[fields[fn] for fn in field_specs.custom_fields[cfn]])

        nonnan_fields = {
            fn: data[mask]
            for fn, data in fields.items()
        }
        return nonnan_fields

    def preprocess(self, fields, cfg):
        for fn, offset in cfg.get('offset', {}).items():
            fields[fn] -= offset

        for fn, scale_factor in cfg.get('scale', {}).items():
            fields[fn] /= scale_factor
        return fields

    @lru_cache(maxsize=None)
    def __getitem__(self, item_idx):
        # if item_idx in self.CACHE:
        #     return self.CACHE[item_idx]

        time_idx, lat_idx, lon_idx = np.unravel_index(item_idx, self.ds_size)
        slice_args = {
            "time_min": self.times[time_idx],
            "time_max": self.times[time_idx + self.ds_cfg.time_window],
            "lat_min": self.lats[lat_idx],
            "lat_max": self.lats[lat_idx + self.ds_cfg.lat_window],
            "lon_min": self.lons[lon_idx],
            "lon_max": self.lons[lon_idx + self.ds_cfg.lon_window],
        }
        item = {}
        for src in self.data_cfg.srcs:
            # print(f"fetching {src} data for {slice_args}")

            if src == 'nadirs':
                field_specs = self.data_cfg.specs['nadirs']
                tmp_nadir = []
                for nadir_src in self.data_cfg.srcs['nadirs']:
                    tmp_nadir.append(
                        self.fetch_fields(slice_args, self.data_cfg.srcs['nadirs'][nadir_src],
                                          field_specs))
                fields = {fn: np.concatenate([nad_f[fn] for nad_f in tmp_nadir], axis=0) for fn in
                          field_specs.fields}
                item['nadirs'] = self.preprocess(fields, self.data_cfg.preprocess.get('nadirs', {}))
            else:
                fields = self.fetch_fields(slice_args, self.data_cfg.srcs[src], self.data_cfg.specs[src])
                item[src] = self.preprocess(fields, self.data_cfg.preprocess.get(src, {}))

        # self.CACHE[item_idx] = item
        return item


def main():
    try:
        ConfigStore.instance().store(
            name=f"raw_dataloading",
            node=Config(),
        )

        with initialize():
            cfg: Config = compose('raw_dataloading')

        # %% Generate data

        # %% Instant data
        pl.seed_everything(42)
        try:
            train_ds
        except:
            train_ds = ConcatDataset([XrDataset(cfg.xp.data, ds_cfg) for ds_cfg in cfg.xp.training.train_ds_cfg])
            val_ds = ConcatDataset([XrDataset(cfg.xp.data, ds_cfg) for ds_cfg in cfg.xp.training.val_ds_cfg])
            test_ds = ConcatDataset([XrDataset(cfg.xp.data, ds_cfg) for ds_cfg in cfg.xp.training.val_ds_cfg])

        # train_dl = DataLoader(dataset=train_ds, batch_size=cfg.xp.training.bs, shuffle=True, num_workers=0)
        train_dl = DataLoader(dataset=val_ds, batch_size=cfg.xp.training.bs, shuffle=True, num_workers=0)
        val_dl = DataLoader(dataset=val_ds, batch_size=cfg.xp.training.bs, shuffle=False, num_workers=0)
        test_dl = DataLoader(dataset=test_ds, batch_size=cfg.xp.training.bs, shuffle=False, num_workers=0)
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()
import uuid
from uuid import uuid4
str(uuid4())

print(f'''
curl 'http://localhost:8888/api/sessions?1644853468714' \\
  -H 'Authorization: token d6e610be39d12c17a8b653633dc66826a0511376ab36648b' \\
  --data-raw '{{"path":"{str(uuid4())}","type":"console","name":"myrepl2","kernel":{{"name":"python3"}}}}' \\
  --compressed
  ''')
