import numpy as np
import pandas as pd
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class XrDataset(Dataset):
    """
    torch Dataset based on an xarray file with on the fly slicing.
    """

    def __init__(self, path, var, slice_win, dim_range=None, strides=None, decode=False,resize_factor=None):
        """
        :param path: xarray file
        :param var: data variable to fetch
        :param slice_win: window size for each dimension {<dim>: <win_size>...}
        :param dim_range: Optional dimensions bounds for each dimension {<dim>: slice(<min>, <max>)...}
        :param strides: strides on each dim while scanning the dataset {<dim>: <dim_stride>...}
        :param decode: Whether to decode the time dim xarray (useful for gt dataset)
        """
        super().__init__()

        self.var = var
        self.resolution = 1
        _ds = xr.open_dataset(path)
        if decode:
            _ds.time.encoding["units"] = "days since 2012-10-01"
            _ds = xr.decode_cf(_ds)
        _ds['time'] = pd.date_range(start="2012-10-01", periods=_ds.sizes['time'], freq='D')
        self.ds = _ds.sel(**(dim_range or {}))
        if resize_factor is not None:
            self.ds = self.ds.coarsen(lon=resize_factor).mean(skipna=True).coarsen(lat=resize_factor).mean(skipna=True)
            self.resolution = self.resolution*resize_factor
        self.slice_win = slice_win
        self.strides = strides or {}
        self.lon = np.arange(dim_range['lon'].start, dim_range['lon'].stop, self.resolution)
        self.lat = np.arange(dim_range['lat'].start, dim_range['lat'].stop, self.resolution)
        self.original_coords = self.ds.coords
        self.padded_coords = self.ds.coords

        self.ds_size = {
            dim: max((self.ds.dims[dim] - slice_win[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in slice_win
        }

    def __del__(self):
        self.ds.close()

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __getitem__(self, item):
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.slice_win[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }
        return self.ds.isel(**sl)[self.var].data.astype(np.float32)


class FourDVarNetDataset(Dataset):
    """
    Dataset for the 4DVARNET method:
        an item contains a slice of OI, mask, and GT
        does the preprocessing for the item
    """

    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            obs_path='/users/local/m19beauc/deep_OI/toy_data/diffusion_dataset.nc',
            obs_var='y',
            gt_path='/users/local/m19beauc/deep_OI/toy_data/diffusion_dataset.nc',
            gt_var='x',
            oi_path=None,
            oi_var=None,
            resize_factor=None,
    ):
        super().__init__()

        self.gt_ds = XrDataset(gt_path, gt_var, slice_win=slice_win, dim_range=dim_range, strides=strides, resize_factor=resize_factor)
        self.obs_ds = XrDataset(obs_path, obs_var, slice_win=slice_win, dim_range=dim_range,strides=strides,resize_factor=resize_factor)
        if oi_var is not None:
            self.oi_ds = XrDataset(oi_path, oi_var, slice_win=slice_win, dim_range=dim_range, strides=strides, resize_factor=resize_factor)
        else:
            self.oi_ds = None
        self.norm_stats = None

    def set_norm_stats(self, stats):
        self.norm_stats = stats

    def coordXY(self):
        return self.gt_ds.lon, self.gt_ds.lat

    def __len__(self):
        return min(len(self.gt_ds), len(self.obs_ds))

    def __getitem__(self, item):
        mean, std = self.norm_stats
        _gt_item = (self.gt_ds[item] - mean) / std
        _oi_item = (self.oi_ds[item] - mean) / std
        _obs_item = self.obs_ds[item] - mean / std
        obs_mask_item = np.where(~np.isnan(_obs_item), 1., 0.)
        gt_item = np.where(~np.isnan(_gt_item), _gt_item, 0.)
        obs_item = np.where(obs_mask_item==1., _gt_item, 0.)
        if self.oi_ds is not None:
            _oi_item = (self.oi_ds[item] - mean) / std
            oi_item = np.where(~np.isnan(_oi_item), _oi_item, 0.)
            return obs_item, obs_mask_item, gt_item, oi_item
        else:
            return obs_item, obs_mask_item, gt_item

class FourDVarNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            train_slices=(slice('2012-10-01', "2012-10-15"),),
            test_slices=(slice('2012-12-30', "2013-01-10"),),
            val_slices=(slice('2012-10-15', "2012-10-30"),),            
            obs_path='/users/local/m19beauc/deep_OI/toy_data/diffusion_dataset.nc',
            obs_var='y',
            gt_path='/users/local/m19beauc/deep_OI/toy_data/diffusion_dataset.nc',
            gt_var='x',
            oi_path=None,
            oi_var=None,
            resize_factor=None,
            dl_kwargs=None):

        super().__init__()
        self.resolution = 1
        self.obs_path = obs_path
        self.obs_var = obs_var
        self.gt_path = gt_path
        self.gt_var = gt_var
        self.oi_path = oi_path
        self.oi_var = oi_var
        self.resize_factor = resize_factor
        self.dim_range = dim_range
        self.slice_win = slice_win
        self.strides = strides
        if self.resize_factor is not None:
            self.slice_win['lon'] = int(self.slice_win['lon']/self.resize_factor)
            self.slice_win['lat'] = int(self.slice_win['lat']/self.resize_factor)
            self.strides['lon'] = int(self.strides['lon']/self.resize_factor)
            self.strides['lat'] = int(self.strides['lat']/self.resize_factor)
        self.dl_kwargs = {
            **{'batch_size': 2, 'num_workers': 2, 'pin_memory': True},
            **(dl_kwargs or {})
        }

        self.train_slices, self.test_slices, self.val_slices = train_slices, test_slices, val_slices
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.norm_stats = None

    def compute_norm_stats(self, ds):
        mean = float(xr.concat([_ds.gt_ds.ds[_ds.gt_ds.var] for _ds in ds.datasets], dim='time').mean())
        std = float(xr.concat([_ds.gt_ds.ds[_ds.gt_ds.var] for _ds in ds.datasets], dim='time').std())
#        mean = 0. ; std = 1.
        return mean, std

    def set_norm_stats(self, ds, ns):
        for _ds in ds.datasets:
            _ds.set_norm_stats(ns)

    def get_domain_bounds(self,ds):
        min_lon = round(np.min(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])),2)
        max_lon = round(np.max(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])),2)
        min_lat = round(np.min(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])),2)
        max_lat = round(np.max(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])),2)       
        return min_lon, max_lon, min_lat, max_lat

    def get_domain_bounds(self, ds):
        min_lon = round(np.min(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        max_lon = round(np.max(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        min_lat = round(np.min(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        max_lat = round(np.max(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        return min_lon, max_lon, min_lat, max_lat

    def coordXY(self):
        return self.test_ds.datasets[0].coordXY()

    def get_padded_coords(self):
        return self.test_ds.datasets[0].gt_ds.padded_coords

    def get_original_coords(self):
        return self.test_ds.datasets[0].gt_ds.original_coords

    def get_domain_split(self):
        return self.test_ds.datasets[0].gt_ds.ds_size

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.test_ds = [
            ConcatDataset(
                [FourDVarNetDataset(
                    dim_range={**self.dim_range, **{'time': sl}},
                    strides=self.strides,
                    slice_win=self.slice_win,
                    obs_path=self.obs_path,
                    obs_var=self.obs_var,
                    gt_path=self.gt_path,
                    gt_var=self.gt_var,
                    oi_path=self.oi_path,
                    oi_var=self.oi_var,
                    resize_factor=self.resize_factor
                ) for sl in slices]
            )
            for slices in (self.train_slices, self.val_slices, self.test_slices)
        ]
        self.norm_stats = self.compute_norm_stats(self.train_ds)
        self.set_norm_stats(self.train_ds, self.norm_stats)
        self.set_norm_stats(self.val_ds, self.norm_stats)
        self.set_norm_stats(self.test_ds, self.norm_stats)
        self.bounding_box = self.get_domain_bounds(self.train_ds)
        self.ds_size = self.get_domain_split()

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.dl_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.dl_kwargs, shuffle=False)

