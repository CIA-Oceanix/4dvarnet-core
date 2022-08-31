import dataloading as dl

path = '/home/clement/DATA/CMEMS/OCEANCOLOUR_MED_BGC_L3_MY_009_143/cmems_obs-oc_med_bgc-optics_my_l3-multi-1km_P1D/GT_2019_full_year_FrMedCoast.nc'
var = 'bbpm443'
slice_win={'lat':120,'lon':200}
strides={'lat':1,'lon':1}
dim_range={'time':slice('2019-01-01','2019-03-31'),'lat':slice(42.38, 43.6),'lon':slice(3., 5.58)}

dm = dl.XrDataset(path=path,var=var,slice_win=slice_win,strides=strides,dim_range=dim_range)
