import utils
import xarray as xr
import traceback


base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    'file_paths={fp}'
]



def run1():
    try:
        paths = dict(
                ose_oi_path= '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc',
                ose_obs_mask_path= '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/dataset_nadir_0d.nc',
                ose_gt_path= '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/validation/dataset_nadir_0d.nc',
                ose_sst_path= '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/sst_CMEMS.nc',
                ose_test_along_track= '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc',)
        
        obs_ds = xr.open_dataset(paths['ose_obs_mask_path'])
        oi_ds = xr.open_dataset(paths['ose_oi_path'])
        obs_ds.isel(time=1).ssh.plot()
        oi_ds.isel(time=1).ssh.plot()
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        fn = run1

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    locals().update(main())

