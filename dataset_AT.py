import numpy as np
import xarray as xr
import pandas as pd
import contextlib
from torch.utils.data import Dataset

from torch.utils.data import DataLoader


def find_pad(sl, st, N):
    k = np.floor(N/st)
    if N>((k*st) + (sl-st)):
        pad = (k+1)*st + (sl-st) - N
    elif N<((k*st) + (sl-st)):
        pad = (k*st) + (sl-st) - N
    else:
        pad = 0
    return int(pad/2), int(pad-int(pad/2))


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
                _ds['time'] = pd.to_datetime(_ds.time)

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
        
        if compute:
            self.ds = self.ds.compute()
       
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
        an item contains a slice of obs, GT and their respective mask
        does the preprocessing for the item
    """

    def __init__(
        self,
        slice_win,
        dim_range=None,
        strides=None,
        obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
        obs_mask_decode=False,
        gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
        gt_var='ssh',
        gt_decode=True,
        sst_path=None,
        sst_var=None,
        sst_decode=True,
        resolution=1/20,
        resize_factor=1,
        use_auto_padding=False,
        compute=False,
        pp='std',
    ):
        super().__init__()
        self.use_auto_padding=use_auto_padding
        self.return_coords = False
        self.pp=pp

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
            )
        else:
            self.sst_ds = None

        self.norm_stats = (0, 1)
        self.norm_stats_sst = (0, 1)

    def set_norm_stats(self, stats, stats_sst=None):
        self.norm_stats = stats
        self.norm_stats_sst = stats_sst

    def __len__(self):
        length = min(len(self.gt_ds), len(self.obs_mask_ds))
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
        
        _obs_item = pp(self.obs_mask_ds[item])
        _gt_item = pp(self.gt_ds[item])
        
        #Obs : data and mask
        obs_mask_item = ~np.isnan(_obs_item)
        obs_item = np.where(~np.isnan(_obs_item), _obs_item, np.zeros_like(_obs_item))
        
        # Ground Truth : data and mask
        gt_mask_item = ~np.isnan(_gt_item)
        gt_item = np.where(~np.isnan(_gt_item), _gt_item, np.zeros_like(_gt_item))

        if self.sst_ds == None:
            return obs_mask_item, gt_mask_item, obs_item.astype(np.float32), gt_item.astype(np.float32) 
        else:
            pp_sst = self.get_pp(self.norm_stats_sst)
            _sst_item = pp_sst(self.sst_ds[item % length])
            sst_item = np.where(~np.isnan(_sst_item), _sst_item, 0.)

            return obs_mask_item, gt_mask_item, obs_item.astype(np.float32), gt_item.astype(np.float32), sst_item.astype(np.float32)