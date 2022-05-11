from dataloading import FourDVarNetDataset, FourDVarNetDataModule, XrDataset
from torch.utils.data import ConcatDataset
import numpy as np


def weight_error(w=0.5):
    def fn(item):
        obs_item = item[2]
        obs_true = item[-1]
        
        new_obs_item = np.where(np.isfinite(obs_true), (obs_true * (1 - w) + obs_item * w), obs_item)
        item[2] = new_obs_item
        return item
    return fn

class FourDVarNetCalDataset(FourDVarNetDataset):
    def __init__(
        self,
        *args,
        obs_target_path='/gpfswork/rech/yrf/commun/CalData/full_cal_obs.nc',
        obs_target_var='swot',
        item_prepro=None,
        **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self.obs_target_ds = XrDataset(obs_target_path, obs_target_var, slice_win=kwargs['slice_win'], dim_range=kwargs['dim_range'],
                                     strides=kwargs['strides'], auto_padding=self.use_auto_padding)
        self.item_prepro = item_prepro

    def __getitem__(self, item):
        
        if self.return_coords:
            return super().__getitem__(item)

        mean, std = self.norm_stats
        obs_target_item = (self.obs_target_ds[item] - mean) / std
        item = [*super().__getitem__(item), obs_target_item]
        if self.item_prepro is not None:
            item = self.item_prepro(item)
        return item


class FourDVarNetCalDataModule(FourDVarNetDataModule):
    def __init__(self, 
            *args,
        obs_target_path='/gpfsscratch/rech/nlu/commun/large/dataset_nadir_0d_swot.nc',
        obs_target_var='ssh_mod',
        item_prepro=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.obs_target_path = obs_target_path
        self.obs_target_var = obs_target_var
        self.item_prepro = item_prepro

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.test_ds = [
            ConcatDataset(
                [FourDVarNetCalDataset(
                    dim_range={**self.dim_range, **{'time': sl}},
                    strides=self.strides,
                    slice_win=self.slice_win,
                    oi_path=self.oi_path,
                    oi_var=self.oi_var,
                    obs_mask_path=self.obs_mask_path,
                    obs_mask_var=self.obs_mask_var,
                    obs_target_path=self.obs_target_path,
                    obs_target_var=self.obs_target_var,
                    gt_path=self.gt_path,
                    gt_var=self.gt_var,
                    sst_path=self.sst_path,
                    sst_var=self.sst_var,
                    item_prepro=self.item_prepro,
                    use_auto_padding=self.use_auto_padding,
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
