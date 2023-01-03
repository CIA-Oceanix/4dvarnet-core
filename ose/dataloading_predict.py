import re
import numpy as np
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pandas as pd
import contextlib
import pyinterp
from sklearn.neighbors import KDTree

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


def find_pad(sl, st, N):
    k = np.floor(N/st)
    if N>((k*st) + (sl-st)):
        pad = (k+1)*st + (sl-st) - N
    elif N<((k*st) + (sl-st)):
        pad = (k*st) + (sl-st) - N
    else:
        pad = 0
    return int(pad/2), int(pad-int(pad/2))

def interpolate_na_2D(da, max_value=100.):
    return (
            da.where(np.abs(da) < max_value, np.nan)
            .pipe(lambda da: da)
            .to_dataframe()
            .interpolate()
            .pipe(xr.Dataset.from_dataframe)
    )

class XrDataset(Dataset):
    """
    torch Dataset based on an xarray file with on the fly slicing.
    """
    def __init__(
        self,
        path,
        var,
        slice_win,
        resolution=1/20,
        dim_range=None,
        strides=None,
        decode=False,
        resize_factor=1,
        compute=False,
        auto_padding=True,
        interp_na=False,
    ):
        """
        :param path: xarray file
        :param var: data variable to fetch
        :param slice_win: window size for each dimension {<dim>: <win_size>...}
        :param resolution: input spatial resolution
        :param dim_range: Optional dimensions bounds for each dimension {<dim>: slice(<min>, <max>)...}
        :param strides: strides on each dim while scanning the dataset {<dim>: <dim_stride>...}
        :param decode: Whether to decode the time dim xarray (useful for gt dataset)
        :param compute: whether to convert dask arrays to xr.DataArray (caution memory)
        """
        super().__init__()
        self.return_coords = False
        self.var = var
        self.resolution = resolution
        self.auto_padding = auto_padding
        self.interp_na = interp_na
        # try/except block for handling both netcdf and zarr files
        try:
            _ds = xr.open_dataset(path)
        except OSError as ex:
            raise ex
            _ds = xr.open_zarr(path)
        if decode:
            if str(_ds.time.dtype) == 'float64':
                _ds.time.attrs["units"] = "seconds since 2012-10-01"
                _ds = xr.decode_cf(_ds)
            else:
                _ds['time'] = pd.to_datetime(_ds.time.values)

        # rename latitute/longitude to lat/lon for consistency
        rename_coords = {}
        if not "lat" in _ds.coords and "latitude" in _ds.coords:
            rename_coords["latitude"] = "lat"
        if not "lon" in _ds.coords and "longitude" in _ds.coords:
            rename_coords["longitude"] = "lon"
        _ds = _ds.rename(rename_coords)


        self.ds = _ds.sel(**(dim_range or {}))
        if resize_factor!=1:
            self.ds = self.ds.coarsen(lon=resize_factor).mean(skipna=True).coarsen(lat=resize_factor).mean(skipna=True)
            self.resolution = self.resolution*resize_factor
        # reshape
        # dimensions
        if not self.auto_padding:
            self.original_coords = self.ds.coords
            self.padded_coords = self.ds.coords

        if self.auto_padding:
            # dimensions
            self.Nt, self.Nx, self.Ny = tuple(self.ds.dims[d] for d in ['time', 'lon', 'lat'])
            # store original input coords for later reconstruction in test pipe
            self.original_coords = self.ds.coords

            # I) first padding x and y inside available DS coords
            pad_x = find_pad(slice_win['lon'], strides['lon'], self.Nx)
            pad_y = find_pad(slice_win['lat'], strides['lat'], self.Ny)
            # get additional data for patch center based reconstruction
            dX = [pad_ *self.resolution for pad_ in pad_x]
            dY = [pad_ *self.resolution for pad_ in pad_y]
            dim_range_ = {
              'lon': slice(self.ds.lon.min().item()-dX[0], self.ds.lon.max().item()+dX[1]),
              'lat': slice(self.ds.lat.min().item()-dY[0], self.ds.lat.max().item()+dY[1]),
              'time': dim_range['time']
            }
            self.ds = _ds.sel(**(dim_range_ or {}))
            self.Nt, self.Nx, self.Ny = tuple(self.ds.dims[d] for d in ['time', 'lon', 'lat'])

            # II) second padding x and y using padding
            pad_x = find_pad(slice_win['lon'], strides['lon'], self.Nx)
            pad_y = find_pad(slice_win['lat'], strides['lat'], self.Ny)
            # pad the dataset
            dX = [pad_ *self.resolution for pad_ in pad_x]
            dY = [pad_ *self.resolution for pad_ in pad_y]
            pad_ = {'lon':(pad_x[0],pad_x[1]),
                    'lat':(pad_y[0],pad_y[1])}

            self.ds_reflected = self.ds.pad(pad_, mode='reflect')
            self.Nx += np.sum(pad_x)
            self.Ny += np.sum(pad_y)

            # compute padded coords end values with linear ramp
            # and replace reflected ones
            end_coords = {
                'lat': (
                    self.ds.lat.values[0]  - pad_['lat'][0] * self.resolution,
                    self.ds.lat.values[-1] + pad_['lat'][1] * self.resolution
                ),
                'lon': (
                    self.ds.lon.values[0]  - pad_['lon'][0] * self.resolution,
                    self.ds.lon.values[-1] + pad_['lon'][1] * self.resolution
                )
            }
            self.padded_coords = {
                c: self.ds[c].pad(pad_, end_values=end_coords, mode="linear_ramp")
                for c in end_coords.keys()
            }

            # re-assign correctly padded coords in place of reflected coords
            self.ds = self.ds_reflected.assign_coords(
                lon=self.padded_coords['lon'], lat=self.padded_coords['lat']
            )
            # III) get lon-lat for the final reconstruction
            dX = ((slice_win['lon']-strides['lon'])/2)*self.resolution
            dY = ((slice_win['lat']-strides['lat'])/2)*self.resolution
            dim_range_ = {
              'lon': slice(dim_range_['lon'].start+dX, dim_range_['lon'].stop-dX),
              'lat': slice(dim_range_['lat'].start+dY, dim_range_['lat'].stop-dY),
            }


        self.ds = self.ds.transpose("time", "lat", "lon")
        if self.interp_na:
            self.ds = interpolate_na_2D(self.ds)

        if compute:
            self.ds = self.ds.compute()

        self.ds = self.ds.transpose("time", "lat", "lon")


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

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @contextlib.contextmanager
    def get_coords(self):
        try:
            self.return_coords = True
            yield
        finally:
            self.return_coords = False

    def __getitem__(self, item):
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.slice_win[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }
        if self.return_coords:
            return self.ds.isel(**sl).coords
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
        oi_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc',
        oi_var='ssh_mod',
        oi_decode=False,
        obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
        obs_mask_decode=False,
        gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
        gt_var='ssh',
        gt_decode=True,
        dim_range_analog=None, 
        analog_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
        analog_var='ssh',
        analog_decode=True,
        sst_path=None,
        sst_var=None,
        sst_decode=True,
        resolution=1/20,
        resize_factor=1,
        use_auto_padding=False,
        aug_train_data=False,
        compute=False,
        pp='std',
    ):
        super().__init__()
        self.use_auto_padding=use_auto_padding

        self.aug_train_data = aug_train_data
        self.return_coords = False
        self.pp=pp

        self.dim_range = dim_range
        self.slice_win = slice_win
        self.strides = strides

        self.gt_ds = XrDataset(
            gt_path, gt_var,
            slice_win=slice_win,
            resolution=resolution,
            dim_range=dim_range,
            strides=strides,
            decode=gt_decode,
            resize_factor=resize_factor,
            compute=compute,
            auto_padding=use_auto_padding,
            interp_na=False,
        )

        self.obs_mask_ds = XrDataset(
            obs_mask_path, obs_mask_var,
            slice_win=slice_win,
            resolution=resolution,
            dim_range=dim_range,
            strides=strides,
            decode=obs_mask_decode,
            resize_factor=resize_factor,
            compute=compute,
            auto_padding=use_auto_padding,
            interp_na=False,
        )

        self.oi_ds = XrDataset(
            oi_path, oi_var,
            slice_win=slice_win,
            resolution=resolution,
            dim_range=dim_range,
            strides=strides,
            decode=oi_decode,
            resize_factor=resize_factor,
            compute=compute,
            auto_padding=use_auto_padding,
            interp_na=True,
        )

        self.analog_ds = XrDataset(
            analog_path, analog_var,
            slice_win=slice_win,
            resolution=resolution,
            dim_range=dim_range_analog,
            strides=strides,
            decode=analog_decode,
            resize_factor=resize_factor,
            compute=compute,
            auto_padding=use_auto_padding,
            interp_na=True,
        )

        if sst_var is not None:
            self.sst_ds = XrDataset(
                sst_path, sst_var,
                slice_win=slice_win,
                resolution=resolution,
                dim_range=dim_range,
                strides=strides,
                decode=sst_decode,
                resize_factor=resize_factor,
                compute=compute,
                auto_padding=use_auto_padding,
                interp_na=False,
            )
        else:
            self.sst_ds = None

        if self.aug_train_data:
            self.perm = np.random.permutation(len(self.obs_mask_ds))

        self.norm_stats = (0, 1)
        self.norm_stats_sst = (0, 1)

        # build analog tree
        # find the n_analog
        self.length_analog = len(self.analog_ds)
        self.xmin, self.xmax = np.min(self.coordXY()[0]), np.max(self.coordXY()[0])
        self.ymin, self.ymax = np.min(self.coordXY()[1]), np.max(self.coordXY()[1])
        self.bin_lon_step = 1
        self.bin_lat_step = 1
        self.binning = pyinterp.Binning2D(
            pyinterp.Axis(np.arange(self.xmin, self.xmax + self.bin_lon_step,
                                    self.bin_lon_step), is_circle=True),
            pyinterp.Axis(np.arange(self.ymin, self.ymax + self.bin_lat_step,
                                    self.bin_lat_step)))
        meshgrid_original = np.meshgrid(self.coordXY()[0],self.coordXY()[1])
        N = self.slice_win['time']
        self.grid_lon, self.grid_lat = np.stack([meshgrid_original[0]]*N), np.stack([meshgrid_original[1]]*N)
        # bins agreggation (10*10) OSSE
        list_binns = []
        for i_analog in range(self.length_analog):
            obs = self.analog_ds[i_analog].flatten()
            self.binning.push(self.grid_lon.flatten(), self.grid_lat.flatten(), obs, True)
            list_binns.append(self.binning.variable('mean'))
            self.binning.clear()
        list_binns = np.stack(list_binns).reshape(self.length_analog,-1)
        # create tree
        self.tree = KDTree(list_binns, leaf_size=50, metric='euclidean')

    def set_norm_stats(self, stats, stats_sst=None):
        self.norm_stats = stats
        self.norm_stats_sst = stats_sst

    def __len__(self):
        length = min(len(self.oi_ds), len(self.gt_ds), len(self.obs_mask_ds))
        if self.aug_train_data:
            factor = int(self.aug_train_data) + 1
            return factor * length
        return length

    def coordXY(self):
        # return self.gt_ds.lon, self.gt_ds.lat
        return self.gt_ds.ds.lon.data, self.gt_ds.ds.lat.data

    @contextlib.contextmanager
    def get_coords(self):
        try:
            self.return_coords = True
            yield
        finally:
            self.return_coords = False

    def get_pp(self, normstats):
        bias, scale = normstats
        return lambda t: (t-bias)/scale


    def __getitem__(self, item):
        if self.return_coords:
            with self.gt_ds.get_coords():
                return self.gt_ds[item]
        pp = self.get_pp(self.norm_stats)
        length = len(self.obs_mask_ds)
        length_analog = len(self.analog_ds)
        if item < length:
            _oi_item = self.oi_ds[item]
            _obs_item = pp(self.obs_mask_ds[item])
            _gt_item = pp(self.gt_ds[item])
        else:
            _oi_item = self.oi_ds[item % length]
            _gt_item = pp(self.gt_ds[item % length])
            nperm  = item // length
            pitem = item % length
            for _ in range(nperm):
                pitem = self.perm[pitem]
            _obs_mask_item = self.obs_mask_ds[pitem]
            obs_mask_item = ~np.isnan(_obs_mask_item)
            _obs_item = np.where(obs_mask_item, _gt_item, np.full_like(_gt_item,np.nan))
 
        obs_mask_item = ~np.isnan(_obs_item)

        # find the n_analog
        # bins agreggation (10*10) observations
        lon_obs = np.where(obs_mask_item,self.grid_lon,np.nan).flatten()
        lat_obs = np.where(obs_mask_item,self.grid_lat,np.nan).flatten()
        obs = np.where(obs_mask_item,_obs_item,np.nan).flatten()
        lon_obs = lon_obs[~np.isnan(lon_obs)]
        lat_obs = lat_obs[~np.isnan(lat_obs)]
        obs = obs[~np.isnan(obs)]
        # bins agreggation (10*10) OSSE
        self.binning.push(lon_obs, lat_obs, obs, True)
        X = np.nan_to_num(self.binning.variable('mean').flatten().reshape(1, -1))
        self.binning.clear()
        # create query (get item)
        dist_knn, index_knn = self.tree.query(X, k=100)
        # retrieve the analogs
        _analog_item = np.stack( [ pp(self.analog_ds[item]) for item in index_knn[0] ], axis=0)
        #_analog_item = np.stack( [ pp(self.analog_ds[item]) for item in np.random.randint(0,length_analog-1,50) ], axis=0)

        _oi_item = pp(np.where(
            np.abs(_oi_item) < 10,
            _oi_item,
            np.nan,
        ))

        # glorys model has NaNs on land
        gt_item = _gt_item
        analog_item = _analog_item

        oi_item = np.where(~np.isnan(_oi_item), _oi_item, 0.)
        # obs_mask_item = self.obs_mask_ds[item].astype(bool) & ~np.isnan(oi_item) & ~np.isnan(_gt_item)

        obs_mask_item = ~np.isnan(_obs_item)
        obs_item = np.where(~np.isnan(_obs_item), _obs_item, np.zeros_like(_obs_item))

        # remove patches from data
        '''
        supervised = False
        if supervised==False:
            n_patch = 10
            s_patch = 10
            for i in range(len(obs_item)):
                posx = np.random.randint(s_patch,self.slice_win['lon']-s_patch,n_patch)
                posy = np.random.randint(s_patch,self.slice_win['lat']-s_patch,n_patch)
                ix = np.stack([np.arange(posx[ipatch]-s_patch,posx[ipatch]+s_patch+1) for ipatch in range(n_patch)])
                iy = np.stack([np.arange(posy[ipatch]-s_patch,posy[ipatch]+s_patch+1) for ipatch in range(n_patch)])
                ix, iy = np.transpose(np.stack([np.meshgrid(ix[ipatch],iy[ipatch]) for ipatch in range(n_patch)]),
                                      (1,0,2,3))
                gt_item[i,ix,iy] = 0.
                obs_item[i,~ix,~iy] = 0.
                obs_mask_item[i,~ix,~iy] = 0.
        '''

        if self.sst_ds == None:
            return oi_item, obs_mask_item, obs_item, gt_item, analog_item
        else:
            pp_sst = self.get_pp(self.norm_stats_sst)
            _sst_item = pp_sst(self.sst_ds[item % length])
            sst_item = np.where(~np.isnan(_sst_item), _sst_item, 0.)
            return oi_item, obs_mask_item, obs_item, gt_item, analog_item, sst_item

class FourDVarNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            predict_slices= (slice('2012-10-01', "2012-11-20"), slice('2013-02-07', "2013-09-30")),
            analog_slices= (slice('2013-01-03', "2013-01-27"),),
            oi_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc',
            oi_var='ssh_mod',
            oi_decode=False,
            obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data/dataset_nadir_0d_swot.nc',
            obs_mask_var='ssh_mod',
            obs_mask_decode=False,
            gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            gt_var='ssh',
            gt_decode=True,
            dim_range_analog=None,
            analog_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            analog_var='ssh',
            analog_decode=True,
            sst_path=None,
            sst_var=None,
            sst_decode=True,
            resize_factor=1,
            aug_train_data=False,
            resolution="1/20",
            dl_kwargs=None,
            compute=False,
            use_auto_padding=False,
            pp='std'
    ):
        super().__init__()
        self.resize_factor = resize_factor
        self.aug_train_data = aug_train_data
        self.dim_range = dim_range
        self.slice_win = slice_win
        self.strides = strides
        self.dl_kwargs = {
            **{'batch_size': 2, 'num_workers': 2, 'pin_memory': True},
            **(dl_kwargs or {})
        }
        self.oi_path = oi_path
        self.oi_var = oi_var
        self.oi_decode = oi_decode
        self.obs_mask_path = obs_mask_path
        self.obs_mask_var = obs_mask_var
        self.obs_mask_decode = obs_mask_decode
        self.gt_path = gt_path
        self.gt_var = gt_var
        self.gt_decode = gt_decode
        self.dim_range_analog = dim_range_analog
        self.analog_path = analog_path
        self.analog_var = analog_var
        self.analog_decode = analog_decode
        self.sst_path = sst_path
        self.sst_var = sst_var
        self.sst_decode = sst_decode

        self.pp=pp
        self.resize_factor = resize_factor
        self.resolution  = parse_resolution_to_float(resolution)
        self.compute = compute
        self.use_auto_padding = use_auto_padding

        self.predict_slices, self.analog_slices =  predict_slices, analog_slices
        self.predict = None, None
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
        return self.predict_ds.datasets[0].coordXY()

    def get_padded_coords(self):
        return self.predict_ds.datasets[0].gt_ds.padded_coords

    def get_original_coords(self):
        return self.predict_ds.datasets[0].gt_ds.original_coords

    def get_domain_split(self):
        return self.predict_ds.datasets[0].gt_ds.ds_size

    def setup(self, stage=None):

        self.predict_ds = ConcatDataset(
                [FourDVarNetDataset(
                    dim_range={**self.dim_range, **{'time': slices[0]}},
                    strides=self.strides,
                    slice_win=self.slice_win,
                    oi_path=self.oi_path,
                    oi_var=self.oi_var,
                    oi_decode=self.oi_decode,
                    obs_mask_path=self.obs_mask_path,
                    obs_mask_var=self.obs_mask_var,
                    obs_mask_decode=self.obs_mask_decode,
                    gt_path=self.gt_path,
                    gt_var=self.gt_var,
                    gt_decode=self.gt_decode,
                    dim_range_analog={**self.dim_range, **{'time': slices[1]}},
                    analog_path=self.analog_path,
                    analog_var=self.analog_var,
                    analog_decode=self.analog_decode,
                    sst_path=self.sst_path,
                    sst_var=self.sst_var,
                    sst_decode=self.sst_decode,
                    resolution=self.resolution,
                    resize_factor=self.resize_factor,
                    compute=self.compute,
                    use_auto_padding=self.use_auto_padding,
                    pp=self.pp,
                )
                for slices in zip(self.predict_slices,self.analog_slices)
             ])

        if self.sst_var is None:
            self.norm_stats = self.compute_norm_stats(self.predict_ds)
            self.set_norm_stats(self.predict_ds, self.norm_stats)
        else:
            self.norm_stats, self.norm_stats_sst = self.compute_norm_stats(self.predict_ds)
            self.set_norm_stats(self.predict_ds, self.norm_stats, self.norm_stats_sst)
        self.bounding_box = self.get_domain_bounds(self.predict_ds)
        self.ds_size = self.get_domain_split()

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, **{**dict(shuffle=False), **self.dl_kwargs})
