import utils
from omegaconf import OmegaConf
import hashlib
from pathlib import Path
import swath_calib.models
import swath_calib.configs
import numpy as np
import pickle
import torch

import traceback


base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    '+datamodule.dl_kwargs.shuffle=False',
    f'file_paths={fp}',
    'params.files_cfg.obs_mask_path=${file_paths.new_noisy_swot}',
    'params.files_cfg.obs_mask_var=five_nadirs',
    'datamodule.aug_train_data=false',
]




def run1():
    try:
        swath_calib.configs.register_configs()
        cal_cfg_n, cal_ckpt = 'ffFalse_swath_calib_qxp20_5nad_no_sst', 'lightning_logs/117_ffFalse_swath_calib_qxp20_5nad_no_sst/checkpoints/last.ckpt'
        cfg = utils.get_cfg(cal_cfg_n)
        cfg_4dvar = utils.get_cfg(cfg.fourdvar_cfg, overrides=overrides)

        cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg_4dvar).encode()).hexdigest()
        saved_data_path = Path('tmp') / f'{cfg_hash}.pk'
        print(saved_data_path)
        rms = lambda da: np.sqrt(np.mean(da**2))
        if saved_data_path.exists():
            print('loading  ', str(saved_data_path))
            with open(saved_data_path, 'rb') as f:
                swath_data = pickle.load(f)
        net = swath_calib.models.build_net(
                in_channels=22,
                out_channels=1,
                **cfg.net_cfg
        )
        normnet = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=22, affine=False, momentum=0.1),
            net
        )

        cal_mod = swath_calib.models.LitDirectCNN(
                # net,
                normnet,
                # gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
                gt_var_stats=[np.array([0]), np.array([1])],
                **cfg.lit_cfg
            )

        print(cal_mod.load_state_dict(torch.load(cal_ckpt, map_location='cpu')['state_dict']))

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
