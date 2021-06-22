import numpy as np
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class XrDataset(Dataset):
    """
    torch Dataset based on an xarray file with on the fly slicing.
    """

    def __init__(self, path, var, slice_win, dim_range=None, strides=None, decode=False):
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
        _ds = xr.open_dataset(path)
        if decode:
            _ds.time.attrs["units"] = "seconds since 2012-10-01"
            _ds = xr.decode_cf(_ds)
        self.ds = _ds.sel(**(dim_range or {}))
        self.slice_win = slice_win
        self.strides = strides or {}
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
            oi_path='/gpfsscratch/rech/nlu/commun/large/ssh_NATL60_swot_4nadir.nc',
            oi_var='ssh_mod',
            obs_mask_path='/gpfsscratch/rech/nlu/commun/large/dataset_nadir_0d_swot.nc',
            obs_mask_var='ssh_mod',
            # obs_mask_var='mask',
            gt_path='/gpfsscratch/rech/nlu/commun/large/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            gt_var='ssh',
            sst_path=None,
            sst_var=None
    ):
        super().__init__()

        self.oi_ds = XrDataset(oi_path, oi_var, slice_win=slice_win, dim_range=dim_range, strides=strides)
        self.gt_ds = XrDataset(gt_path, gt_var, slice_win=slice_win, dim_range=dim_range, strides=strides, decode=True)
        self.obs_mask_ds = XrDataset(obs_mask_path, obs_mask_var, slice_win=slice_win, dim_range=dim_range,
                                     strides=strides)

        self.norm_stats = None

        if sst_var == 'sst':
            self.sst_ds = XrDataset(sst_path, sst_var, slice_win=slice_win, dim_range=dim_range, strides=strides,
                                    decode=True)
        else:
            self.sst_ds = None
        self.norm_stats_sst = None

    def set_norm_stats(self, stats, stats_sst=None):
        self.norm_stats = stats
        self.norm_stats_sst = stats_sst

    def __len__(self):
        return min(len(self.oi_ds), len(self.gt_ds), len(self.obs_mask_ds))

    def __getitem__(self, item):
        mean, std = self.norm_stats
        _oi_item = self.oi_ds[item]
        _oi_item = (np.where(
            np.abs(_oi_item) < 10,
            _oi_item,
            np.nan,
        ) - mean) / std
        _gt_item = (self.gt_ds[item] - mean) / std
        oi_item = np.where(~np.isnan(_oi_item), _oi_item, 0.)
        # obs_mask_item = self.obs_mask_ds[item].astype(bool) & ~np.isnan(oi_item) & ~np.isnan(_gt_item)
        obs_mask_item = ~np.isnan(self.obs_mask_ds[item])

        gt_item = _gt_item

        if self.sst_ds == None:
            return oi_item, obs_mask_item, gt_item
        else:
            mean, std = self.norm_stats_sst
            _sst_item = (self.sst_ds[item] - mean) / std
            sst_item = np.where(~np.isnan(_sst_item), _sst_item, 0.)

            return oi_item, obs_mask_item, gt_item, sst_item


class FourDVarNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            train_slices=(slice('2012-10-01', "2012-11-20"), slice('2013-02-08', "2013-09-30")),
            # train_slices=(slice('2012-10-01', "2012-10-10"),),
            test_slices=(slice('2012-12-30', "2013-01-19"),),
            val_slices=(slice('2012-11-30', "2012-12-20"),),
            oi_path='/gpfsscratch/rech/nlu/commun/large/ssh_NATL60_swot_4nadir.nc',
            oi_var='ssh_mod',
            obs_mask_path='/gpfsscratch/rech/nlu/commun/large/dataset_nadir_0d_swot.nc',
            obs_mask_var='ssh_mod',
            # obs_mask_var='mask',
            gt_path='/gpfsscratch/rech/nlu/commun/large/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            gt_var='ssh',
            sst_path=None,
            sst_var=None,
            dl_kwargs=None,
    ):
        super().__init__()
        self.slice_win = slice_win
        self.dim_range = dim_range
        self.strides = strides
        self.dl_kwargs = {
            **{'batch_size': 8, 'num_workers': 2, 'pin_memory': True},
            **(dl_kwargs or {})
        }

        self.oi_path = oi_path
        self.oi_var = oi_var
        self.obs_mask_path = obs_mask_path
        self.obs_mask_var = obs_mask_var
        self.gt_path = gt_path
        self.gt_var = gt_var
        self.sst_path = sst_path
        self.sst_var = sst_var

        self.train_slices, self.test_slices, self.val_slices = train_slices, test_slices, val_slices
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.norm_stats = None
        self.norm_stats_sst = None

    def compute_norm_stats(self, ds):
        mean = float(xr.concat([_ds.gt_ds.ds[_ds.gt_ds.var] for _ds in ds.datasets], dim='time').mean())
        std = float(xr.concat([_ds.gt_ds.ds[_ds.gt_ds.var] for _ds in ds.datasets], dim='time').std())

        if self.sst_var == None:
            return mean, std
        else:
            print('... Use SST data')
            mean_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').mean())
            std_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').std())

            return [mean, std], [mean_sst, std_sst]

    def set_norm_stats(self, ds, ns, ns_sst=None):
        for _ds in ds.datasets:
            _ds.set_norm_stats(ns, ns_sst)

    def get_domain_bounds(self, ds):
        min_lon = round(np.min(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        max_lon = round(np.max(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        min_lat = round(np.min(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        max_lat = round(np.max(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        return min_lon, max_lon, min_lat, max_lat

    def get_domain_split(self):
        return self.test_ds.datasets[0].gt_ds.ds_size

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.test_ds = [
            ConcatDataset(
                [FourDVarNetDataset(
                    dim_range={**self.dim_range, **{'time': sl}},
                    strides=self.strides,
                    slice_win=self.slice_win,
                    oi_path=self.oi_path,
                    oi_var=self.oi_var,
                    obs_mask_path=self.obs_mask_path,
                    obs_mask_var=self.obs_mask_var,
                    gt_path=self.gt_path,
                    gt_var=self.gt_var,
                    sst_path=self.sst_path,
                    sst_var=self.sst_var
                ) for sl in slices]
            )
            for slices in (self.train_slices, self.val_slices, self.test_slices)
        ]

        if self.sst_var == None:
            self.norm_stats = self.compute_norm_stats(self.train_ds)
            self.set_norm_stats(self.train_ds, self.norm_stats)
            self.set_norm_stats(self.val_ds, self.norm_stats)
            self.set_norm_stats(self.test_ds, self.norm_stats)
        else:
            self.norm_stats, self.norm_stats_sst = self.compute_norm_stats(self.train_ds)

            self.set_norm_stats(self.train_ds, self.norm_stats, self.norm_stats_sst)
            self.set_norm_stats(self.val_ds, self.norm_stats, self.norm_stats_sst)
            self.set_norm_stats(self.test_ds, self.norm_stats, self.norm_stats_sst)

        self.bounding_box = self.get_domain_bounds(self.train_ds)
        self.ds_size = self.get_domain_split()

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.dl_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.dl_kwargs, shuffle=False)


if __name__ == '__main__':
    """
    Test run for single batch loading and trainer.fit 
    """

    # Specify the dataset spatial bounds
    dim_range = {
        'lat': slice(35, 45),
        'lon': slice(-65, -55),
    }

    # Specify the batch patch size
    slice_win = {
        'time': 5,
        'lat': 200,
        'lon': 200,
    }
    # Specify the stride between two patches
    strides = {
        'time': 1,
        'lat': 200,
        'lon': 200,
    }

    dm = FourDVarNetDataModule(
        slice_win=slice_win,
        dim_range=dim_range,
        strides=strides,
    )

    # Test a single batch loading
    dm.setup()
    dl = dm.val_dataloader()
    batch = next(iter(dl))
    oi, mask, gt = batch

    # Test fit
    from main import LitModel

    lit_mod = LitModel()
    trainer = pl.Trainer(gpus=1)
    # dm.setup()
    trainer.fit(lit_mod, datamodule=dm)
