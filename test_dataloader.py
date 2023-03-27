import dataloading_joint as dl
import matplotlib.pyplot as plt
import numpy as np

dm = dl.FourDVarNetDataModule(
            slice_win={'lat':240,'lon':240,'time':29},
            dim_range={'lat':slice('41.17', '43.6'), 'lon':slice('3.', '6.09')},
            strides={'lat':20,'lon':20,'time':1},
            train_slices= (slice('2019-01-01', '2019-10-10'), slice('2019-11-23', '2019-12-31')),
            test_slices= (slice('2021-02-09', '2021-06-30'),),
            val_slices= (slice('2020-02-04', '2020-03-31'),),
            oi_path='/home/administrateur/4DVarNet/DATA/CMEMS/OCEANCOLOUR_MED_BGC_L3_MY_009_143/cmems_obs-oc_med_bgc-optics_my_l3-multi-1km_P1D/GT_FrMedCoast_log10_2019_2020_2021_.nc',
            oi_var='bbp443',
            obs_mask_path='/home/administrateur/4DVarNet/DATA/CMEMS/OCEANCOLOUR_MED_BGC_L3_MY_009_143/cmems_obs-oc_med_bgc-optics_my_l3-multi-1km_P1D/Obs_patch50_FrMedCoast_log10_2019_2020_2021_.nc',
            obs_mask_var='bbp443',
            gt_path='/home/administrateur/4DVarNet/DATA/CMEMS/OCEANCOLOUR_MED_BGC_L3_MY_009_143/cmems_obs-oc_med_bgc-optics_my_l3-multi-1km_P1D/GT_FrMedCoast_log10_2019_2020_2021_.nc',
            gt_var='bbp443',
            sst_path='/home/administrateur/4DVarNet/DATA/CMEMS/OCEANCOLOUR_MED_BGC_L3_MY_009_143/cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D/GT_FrMedCoast_log10_2019_2020_2021_.nc',
            sst_var='chl',
            sst_mask_path='/home/administrateur/4DVarNet/DATA/CMEMS/OCEANCOLOUR_MED_BGC_L3_MY_009_143/cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D/Obs_patch50_FrMedCoast_log10_2019_2020_2021_.nc',
            sst_mask_var='chl',
            resize_factor=1)

dm.setup()

_, inputs_Mask, inputs_obs, targets_GT, inputs_Mask_sst, inputs_obs_sst, targets_GT_sst = next(iter(dm.train_dataloader()))


plt.figure()
plt.subplot(231)
plt.pcolormesh(inputs_Mask[0,0])
plt.colorbar()
plt.subplot(232)
plt.pcolormesh(inputs_obs[0,0])
plt.colorbar()
plt.subplot(233)
plt.pcolormesh(targets_GT[0,0])
plt.colorbar()
plt.subplot(234)
plt.pcolormesh(inputs_Mask_sst[0,0])
plt.colorbar()
plt.subplot(235)
plt.pcolormesh(inputs_obs_sst[0,0])
plt.colorbar()
plt.subplot(236)
plt.pcolormesh(targets_GT_sst[0,0])
plt.colorbar()
plt.show()
