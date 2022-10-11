import re
import numpy as np
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pandas as pd
import contextlib


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
        
        print('.... path '+path)
        print('....  var: :'+var,flush=True)
        print('.... resize factor: %d'%resize_factor)
        print('.... auto padding %d'%self.auto_padding)
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

        # reshape
        # dimensions
        self.ds = _ds.sel(**(dim_range or {}))
        if resize_factor!=1:
            self.ds = self.ds.coarsen(lon=resize_factor).mean(skipna=True).coarsen(lat=resize_factor).mean(skipna=True)
            self.resolution = self.resolution*resize_factor         

        print('... ds shape %dx%dx%d ' %(self.ds.coords['time'].shape[0],self.ds.coords['lon'].shape[0],self.ds.coords['lat'].shape[0]))             
        
        # reshape
        # dimensions
        if not self.auto_padding:
            self.original_coords = self.ds.coords
            self.padded_coords = self.ds.coords

        if self.auto_padding:                
            # dimensions
            self.ds = _ds.sel(**(dim_range or {}))
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
            self.ds = interpolate_na_2D(self.ds,max_value=500.)

        if compute:
            self.ds = self.ds.compute()
        
        self.slice_win = slice_win
        self.strides = strides or {}
        self.ds_size = {
                dim: max((self.ds.dims[dim] - slice_win[dim]) // self.strides.get(dim, 1) + 1, 0)
                for dim in slice_win
        }
        print('... ds shape %dx%dx%d ' %(self.ds.coords['time'].shape[0],self.ds.coords['lon'].shape[0],self.ds.coords['lat'].shape[0]))             
        print('... slicing window: %dx%dx%d'%(self.slice_win['time'],self.slice_win['lon'],self.slice_win['lat']))

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
        sst_path=None,
        sst_var=None,
        sst_decode=True,
        u_path=None,
        v_path=None,
        u_var=None,
        v_var=None,
        uv_decode=True,
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
            interp_na=True,
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

        if self.aug_train_data > 0 :
            _len = len(self.obs_mask_ds)
            self.perm = np.random.permutation(_len)
            for _ in range(1,self.aug_train_data):
                self.perm = np.concatenate((self.perm,np.random.permutation(_len)),axis=0)
            print('.... Data augmentation : %d/%d'%(len(self.perm),_len))
            
            
        if u_var is not None:
            self.u_ds = XrDataset(
                u_path, u_var,
                slice_win=slice_win,
                resolution=resolution,
                dim_range=dim_range,
                strides=strides,
                decode=uv_decode,
                resize_factor=resize_factor,
                compute=compute,
                auto_padding=use_auto_padding,
                interp_na=True,
            )
        else:
            self.u_ds = None

        if v_var is not None:
            self.v_ds = XrDataset(
                v_path, v_var,
                slice_win=slice_win,
                resolution=resolution,
                dim_range=dim_range,
                strides=strides,
                decode=uv_decode,
                resize_factor=resize_factor,
                compute=compute,
                auto_padding=use_auto_padding,
                interp_na=True,
            )
        else:
            self.v_ds = None

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
                interp_na=True,
            )
        else:
            self.sst_ds = None
        

        self.norm_stats = (0, 1)
        self.norm_stats_sst = (0, 1)
        self.norm_stats_uv = (0, 1)

    def set_norm_stats(self, stats, stats_sst=None,stats_uv=None):
        self.norm_stats = stats
        self.norm_stats_sst = stats_sst
        self.norm_stats_uv = stats_uv

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
        
        # retrieve lat,lon        
        pp = self.get_pp(self.norm_stats)
        length = len(self.obs_mask_ds)
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

        _oi_item = pp(np.where(
            np.abs(_oi_item) < 10,
            _oi_item,
            np.nan,
        ))

        # glorys model has NaNs on land
        gt_item = _gt_item

        oi_item = np.where(~np.isnan(_oi_item), _oi_item, 0.)
        # obs_mask_item = self.obs_mask_ds[item].astype(bool) & ~np.isnan(oi_item) & ~np.isnan(_gt_item)

        obs_mask_item = ~np.isnan(_obs_item)
        obs_item = np.where(~np.isnan(_obs_item), _obs_item, np.zeros_like(_obs_item))

        if self.sst_ds == None:
            return oi_item, obs_mask_item, obs_item, gt_item
        else:
            pp_sst = self.get_pp(self.norm_stats_sst)
            _sst_item = pp_sst(self.sst_ds[item % length])
            sst_item = np.where(~np.isnan(_sst_item), _sst_item, 0.)

            if self.u_ds == None:    
                return oi_item, obs_mask_item, obs_item, gt_item, sst_item
            else:
                pp_uv = self.get_pp(self.norm_stats_uv)
                
                _u_item = pp_uv(self.u_ds[item % length])
                u_item = np.where(~np.isnan(_u_item), _u_item, 0.)
                
                _v_item = pp_uv(self.v_ds[item % length])
                v_item = np.where(~np.isnan(_v_item), _v_item, 0.)
    
                
                if 1*1 :
                    with self.gt_ds.get_coords():
                        _item_coords = self.gt_ds[item % length]

                        lat_item = np.float32( _item_coords['lat'] )
                        lon_item = np.float32( _item_coords['lon'] )

                    return oi_item, obs_mask_item, obs_item, gt_item, sst_item, u_item, v_item, lat_item, lon_item
                else:          
                    return oi_item, obs_mask_item, obs_item, gt_item, sst_item, u_item, v_item

class FourDVarNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            train_slices= (slice('2012-10-01', "2012-11-20"), slice('2013-02-07', "2013-09-30")),
            test_slices= (slice('2013-01-03', "2013-01-27"),),
            val_slices= (slice('2012-11-30', "2012-12-24"),),
            oi_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc',
            oi_var='ssh_mod',
            oi_decode=False,
            obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data/dataset_nadir_0d_swot.nc',
            obs_mask_var='ssh_mod',
            obs_mask_decode=False,
            gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            gt_var='ssh',
            gt_decode=True,
            sst_path=None,
            sst_var=None,
            sst_decode=True,
            u_path=None,
            u_var=None,
            v_path=None,
            v_var=None,
            uv_decode=True,
            resize_factor=1,
            aug_train_data=0,
            resolution="1/20",
            dl_kwargs=None,
            compute=False,
            use_auto_padding=False,
            pp='std'
    ):
        super().__init__()
        
        print('.... resize_factor %d'%resize_factor,flush=True)
        print('')
        
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
        self.sst_path = sst_path
        self.sst_var = sst_var
        self.sst_decode = sst_decode
        self.u_path = u_path
        self.u_var = u_var
        self.v_path = v_path
        self.v_var = v_var
        self.uv_decode = uv_decode

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
            
            print('..... sst data = %f -- %f'%(mean_sst,std_sst) )
            if self.u_var == None:

                return [mean, std], [mean_sst, std_sst]
            else:
                print('... Use (U,V) data')
                mean_uv = 0.
                var_u = float(xr.concat([_ds.u_ds.ds[_ds.u_ds.var]**2 for _ds in ds.datasets], dim='time').mean())
                var_v = float(xr.concat([_ds.v_ds.ds[_ds.v_ds.var]**2 for _ds in ds.datasets], dim='time').mean())
                
                std_uv = np.sqrt(var_u + var_v)

                return [mean, std], [mean_sst, std_sst], [mean_uv, std_uv]

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
            if self.u_var == None:
    
                return [m, M-m], [m_sst, M_sst-m_sst]
            else:
                print('... Use (U,V) data')
                m_u = float(xr.concat([_ds.u_ds.ds[_ds.u_ds.var] for _ds in ds.datasets], dim='time').min())
                M_u = float(xr.concat([_ds.u_ds.ds[_ds.u_ds.var] for _ds in ds.datasets], dim='time').max())
                m_v = float(xr.concat([_ds.v_ds.ds[_ds.v_ds.var] for _ds in ds.datasets], dim='time').min())
                M_v = float(xr.concat([_ds.v_ds.ds[_ds.v_ds.var] for _ds in ds.datasets], dim='time').max())

                m_uv = np.min([m_u,m_v])                
                M_uv = np.max([M_u,M_v])                

                return [m, M-m], [m_sst, M_sst-m_sst], [m_uv, M_uv-m_uv]

    def compute_norm_stats(self, ds):
        if self.pp == 'std':
            return self.mean_stds(ds)
        elif self.pp == 'norm':
            return self.min_max(ds)

    def compute_scaling_uv_geo(self,ds,sigma=4.):
        from scipy import ndimage
        from scipy.ndimage import gaussian_filter
        
        dssh_dy_u = 0.
        dssh_dx_v = 0.
        dssh_dy_v = 0.
        dssh_dx_u = 0.
        norm_dx = 0.
        norm_dy = 0.
        norm_u = 0.
        norm_v = 0.
        
        for _ds in ds.datasets:
            
            # get fields
            ssh = _ds.gt_ds.ds[_ds.gt_ds.var]
            u  = _ds.u_ds.ds[_ds.u_ds.var]
            v  = _ds.v_ds.ds[_ds.v_ds.var]
 
            #print( u.shape )
            #print( ssh.shape )
            #print( _ds.sst_ds.ds[_ds.sst_ds.var] )
            
 
            if 1*0 :
                ssh = ssh[3:43,20:220,20:220]
                u = u[3:43,20:220,20:220]
                v = v[3:43,20:220,20:220]
            else:
                ssh = ssh[:,20:220,20:220]
                u = u[:,20:220,20:220]
                v = v[:,20:220,20:220]

            #print( np.sum( np.isnan(ssh) ) )
            #print( u.shape )
            #print( v.shape )
            
            #print( np.mean( ssh**2 ) )
            
            # Gaussian filtering
            u = gaussian_filter(u, sigma=sigma)
            v = gaussian_filter(v, sigma=sigma)
            ssh = gaussian_filter(ssh, sigma=sigma)
                        
            # ssh gradients
            dssh_dx = 1. * ndimage.sobel(ssh,axis=2)
            dssh_dy = 1. * ndimage.sobel(ssh,axis=1)   

            print( u.shape )
            print( ssh.shape )

            w = np.isnan( u + v + dssh_dy + dssh_dx ).astype(float)
            
            u = u[ w == False ] 
            v = v[ w == False ] 
            dssh_dx = dssh_dx[ w == False ]
            dssh_dy = dssh_dy[ w == False ]
                    
            dssh_dy_u += np.sum( -1. * dssh_dy * u )
            dssh_dy_v += np.sum( -1. * dssh_dy * v )

            dssh_dx_v += np.sum( 1. * dssh_dx * v )
            dssh_dx_u += np.sum( 1. * dssh_dx * u )

            norm_dy += np.sum( dssh_dy ** 2 )
            norm_dx += np.sum( dssh_dx ** 2 )
            
            norm_u +=  np.sum( u ** 2)
            norm_v +=  np.sum( v ** 2)
                
        alpha_dy_u = dssh_dy_u / norm_dy
        alpha_dx_v = dssh_dx_v / norm_dx
        
        corr_dy_u = dssh_dy_u / np.sqrt( norm_dy * norm_u  )
        corr_dy_v = dssh_dy_v / np.sqrt( norm_dy * norm_v  )
        corr_dx_u = dssh_dx_u / np.sqrt( norm_dx * norm_u  )
        corr_dx_v = dssh_dx_v / np.sqrt( norm_dx * norm_v  )
       
        print('... R**2: %f -- %f --  %f -- %f'%(corr_dx_u,corr_dx_v,corr_dy_u,corr_dy_v))
        print('.... alpha: %f -- %f -- %f'%(alpha_dx_v,alpha_dy_u,alpha_dy_u/alpha_dx_v)  )

        return 1.,alpha_dy_u/alpha_dx_v,alpha_dx_v

    def set_norm_stats(self, ds, ns, ns_sst=None,ns_uv=None):
        for _ds in ds.datasets:
            _ds.set_norm_stats(ns, ns_sst, ns_uv)

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
                oi_path=self.oi_path,
                oi_var=self.oi_var,
                oi_decode=self.oi_decode,
                obs_mask_path=self.obs_mask_path,
                obs_mask_var=self.obs_mask_var,
                obs_mask_decode=self.obs_mask_decode,
                gt_path=self.gt_path,
                gt_var=self.gt_var,
                gt_decode=self.gt_decode,
                sst_path=self.sst_path,
                sst_var=self.sst_var,
                sst_decode=self.sst_decode,
                u_path=self.u_path,
                u_var=self.u_var,
                v_path=self.v_path,
                v_var=self.v_var,
                uv_decode=self.uv_decode,
                resolution=self.resolution,
                resize_factor=self.resize_factor,
                aug_train_data=self.aug_train_data,
                compute=self.compute,
                pp=self.pp,
            ) for sl in self.train_slices])


        self.val_ds, self.test_ds = [
            ConcatDataset(
                [FourDVarNetDataset(
                    dim_range={**self.dim_range, **{'time': sl}},
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
                    sst_path=self.sst_path,
                    sst_var=self.sst_var,
                    sst_decode=self.sst_decode,
                    u_path=self.u_path,
                    u_var=self.u_var,
                    v_path=self.v_path,
                    v_var=self.v_var,
                    uv_decode=self.uv_decode,
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
            if self.u_var is None:
                self.norm_stats, self.norm_stats_sst = self.compute_norm_stats(self.train_ds)
                self.set_norm_stats(self.train_ds, self.norm_stats, self.norm_stats_sst)
                self.set_norm_stats(self.val_ds, self.norm_stats, self.norm_stats_sst)
                self.set_norm_stats(self.test_ds, self.norm_stats, self.norm_stats_sst)
            else:
                self.norm_stats, self.norm_stats_sst, self.norm_stats_uv = self.compute_norm_stats(self.train_ds)
                self.set_norm_stats(self.train_ds, self.norm_stats, self.norm_stats_sst, self.norm_stats_uv)
                self.set_norm_stats(self.val_ds, self.norm_stats, self.norm_stats_sst, self.norm_stats_uv)
                self.set_norm_stats(self.test_ds, self.norm_stats, self.norm_stats_sst, self.norm_stats_uv)

            #self.alpha_dx,self.alpha_dy,self.alpha_uv_geo = self.compute_scaling_uv_geo(self.train_ds)
        
        self.bounding_box = self.get_domain_bounds(self.train_ds)
        self.ds_size = self.get_domain_split()
    #def get_scaling_ssh_uv(self):
    #    return self.alpha_dx,self.alpha_dy,self.alpha_uv_geo
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, **{**dict(shuffle=True), **self.dl_kwargs})

    def val_dataloader(self):
        return DataLoader(self.val_ds, **{**dict(shuffle=False), **self.dl_kwargs})

    def test_dataloader(self):
        return DataLoader(self.test_ds, **{**dict(shuffle=False), **self.dl_kwargs})


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
    from utils import get_dm
    dm = get_dm('xp_aug/xp_repro/full_core_sst', add_overrides=['datamodule.sst_path=${file_paths.natl_sst_daily}'])
    
    dl = dm.test_dataloader()
    ds = dl.dataset.datasets[0]
    len(ds.perm)
    batch= next(iter(dl))
    oi, msk, obs, gt, sst_gt, u_gt,v_gt = ds[2]
    oi_, msk_, obs_, gt_, sst_gt_, u_gt_,v_gt_ = ds[2+ len(ds.perm)]
    import matplotlib.pyplot as plt
    ds.sst_ds.ds.isel(time=0).sst.plot()
    p = lambda t: plt.imshow(t[0])
    ds.sst_ds.ds.sst.isel(time=0).plot()
    ds.gt_ds.ds.ssh.isel(time=0).plot()
    ds.obs_mask_ds.ds.ssh_mod.isel(time=5).plot()
    ds.obs_mask_ds.ds.ssh_mod.pipe(np.isfinite).mean('time').isel(lat=slice(20, -20), lon=slice(20,-20)).plot()
    sst_ds = xr.open_dataset(dm.sst_path)
    dm.dim_range
    _ds = sst_ds
    _ds = _ds.sel(**dm.dim_range)
    _ds.time.attrs["units"] = "seconds since 2012-10-01"
    _ds = xr.decode_cf(_ds)
    _ds = interpolate_na_2D(_ds, max_value=10**10)
    _ds.sel(**dm.dim_range).isel(time=0).sst.plot()
    p(sst_gt)
    p(oi)
    sst_ds.sel(**dm.dim_range).isel(time=0).sst.plot()
