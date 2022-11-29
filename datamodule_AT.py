import re
import numpy as np
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import ConcatDataset, DataLoader
import print_log
from dataset_AT import FourDVarNetDataset

log = print_log.get_logger(__name__)

def parse_resolution_to_float(frac):
    """ Matches a string consting of an integer followed by either a divisor
    ("/" and an integer) or some spaces and a simple fraction (two integers
    separated by "/").
    From https://stackoverflow.com/a/1806375
    Args:
        frac (str): resolution as string fraction or integer from config file
    Returns:
        float: resolution as float
    Example:
        for x in ['3', '1/12', '1/20', '1 2/3']: print(repr(parse_resolution_to_float(x)))
        >
        3.0
        0.0833333333333333
        0.05
        1.6666666666666665
    """
    frac_regex = re.compile(r'^(\d+)(?:(?:\s+(\d+))?/(\d+))?$')
    i, n, d = frac_regex.match(frac).groups()
    if d is None: n, d = 0, 1  # if d is None, then n is also None
    if n is None: i, n = 0, i
    return float(i) + float(n) / float(d)


class FourDVarNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            train_slices= (slice('2012-10-01', "2012-11-20"), slice('2013-02-07', "2013-09-30")),
            test_slices= (slice('2013-01-03', "2013-01-27"),),
            val_slices= (slice('2012-11-30', "2012-12-24"),),
            obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data/dataset_nadir_0d_swot.nc',
            obs_mask_var='ssh_mod',
            obs_mask_decode=False,
            gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            gt_var='ssh',
            gt_decode=True,
            sst_path=None,
            sst_var=None,
            sst_decode=True,
            resize_factor=1,
            resolution="1/20",
            dl_kwargs=None,
            compute=False,
            use_auto_padding=False,
            pp='std'
    ):
        super().__init__()
        self.resize_factor = resize_factor
        self.dim_range = dim_range
        self.slice_win = slice_win
        self.strides = strides
        log.info('Slice win = {} and Strides = {}'.format(self.slice_win, self.strides))
        self.dl_kwargs = {
            **{'batch_size': 2, 'num_workers': 2, 'pin_memory': True},
            **(dl_kwargs or {})
        }
        self.obs_mask_path = obs_mask_path
        self.obs_mask_var = obs_mask_var
        self.obs_mask_decode = obs_mask_decode
        self.gt_path = gt_path
        self.gt_var = gt_var
        self.gt_decode = gt_decode
        self.sst_path = sst_path
        self.sst_var = sst_var
        self.sst_decode = sst_decode

        self.pp=pp
        self.resize_factor = resize_factor
        self.resolution  = parse_resolution_to_float(resolution)
        self.compute = compute
        self.use_auto_padding = use_auto_padding

        self.train_slices, self.test_slices, self.val_slices = train_slices, test_slices, val_slices
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.norm_stats = (0, 1)
        self.norm_stats_sst = None


    def mean_stds(self, ds):
        sum = 0
        count = 0
        for gt in [_it for _ds in ds.datasets for _it in _ds.gt_ds]:
            sum += np.nansum(gt)
            count += np.sum(~np.isnan(gt))
        mean = sum / count
        sum = 0
        for gt in [_it for _ds in ds.datasets for _it in _ds.gt_ds]:
            sum += np.nansum((gt - mean)**2)
        std = (sum / count)**0.5

        if self.sst_var == None:
            return mean, std
        else:
            print('... Use SST data')
            mean_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').mean())
            std_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').std())

            return [mean, std], [mean_sst, std_sst]

    def min_max(self, ds):
        M = -np.inf
        m = np.inf
        for gt in [_it for _ds in ds.datasets for _it in _ds.gt_ds]:
            M =  max(M ,np.nanmax(gt))
            m =  min(m ,np.nanmin(gt))
        if self.sst_var == None:
            return m, M-m
        else:
            print('... Use SST data')
            m_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').min())
            M_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').max())

            return [m, M-m], [m_sst, M_sst-m_sst]

    def compute_norm_stats(self, ds):
        if self.pp == 'std':
            return self.mean_stds(ds)
        elif self.pp == 'norm':
            return self.min_max(ds)

    def set_norm_stats(self, ds, ns, ns_sst=None):
        for _ds in ds.datasets:
            _ds.set_norm_stats(ns, ns_sst)

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
        self.train_ds = ConcatDataset(
            [FourDVarNetDataset(
                dim_range={**self.dim_range, **{'time': sl}},
                strides=self.strides,
                slice_win=self.slice_win,
                obs_mask_path=self.obs_mask_path,
                obs_mask_var=self.obs_mask_var,
                obs_mask_decode=self.obs_mask_decode,
                gt_path=self.gt_path,
                gt_var=self.gt_var,
                gt_decode=self.gt_decode,
                sst_path=self.sst_path,
                sst_var=self.sst_var,
                sst_decode=self.sst_decode,
                resolution=self.resolution,
                resize_factor=self.resize_factor,
                compute=self.compute,
                pp=self.pp,
            ) for sl in self.train_slices])
       
        self.val_ds, self.test_ds = [
            ConcatDataset(
                [FourDVarNetDataset(
                    dim_range={**self.dim_range, **{'time': sl}},
                    strides=self.strides,
                    slice_win=self.slice_win,
                    obs_mask_path=self.obs_mask_path,
                    obs_mask_var=self.obs_mask_var,
                    obs_mask_decode=self.obs_mask_decode,
                    gt_path=self.gt_path,
                    gt_var=self.gt_var,
                    gt_decode=self.gt_decode,
                    sst_path=self.sst_path,
                    sst_var=self.sst_var,
                    sst_decode=self.sst_decode,
                    resolution=self.resolution,
                    resize_factor=self.resize_factor,
                    compute=self.compute,
                    use_auto_padding=self.use_auto_padding,
                    pp=self.pp,
                ) for sl in slices]
            )
            for slices in (self.val_slices, self.test_slices)
        ]

        if self.sst_var is None:
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
        return DataLoader(self.train_ds, **{**dict(shuffle=True), **self.dl_kwargs})

    def val_dataloader(self):
        return DataLoader(self.val_ds, **{**dict(shuffle=False), **self.dl_kwargs})

    def test_dataloader(self):
        return DataLoader(self.test_ds, **{**dict(shuffle=False), **self.dl_kwargs})
