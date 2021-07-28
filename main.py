#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:59:23 2020
@author: rfablet
"""
import os

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

import solver as NN_4DVar
from lit_model_stochastic import LitModelStochastic
# from models import Gradient_img, LitModel, LitModelWithSST
# from new_dataloading import FourDVarNetDataModule
import models
import new_dataloading
from old_dataloading import LegacyDataLoading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gradient_img = models.Gradient_img()


class FourDVarNetRunner:
    def __init__(self, dataloading="old", config=None):
        self.filename_chkpt = 'modelSLAInterpGF-Exp3-{epoch:02d}-{val_loss:.2f}'
        if config is None:
            import config
        else:
            config = __import__("config_" + str(config))

        self.cfg = OmegaConf.create(config.params)
        shape_state = self.cfg.shape_state
        self.wLoss = [0] * self.cfg.dT
        self.wLoss[int(self.cfg.dT / 2)] = 1.
        dataloading = config.params['dataloading']
        print(dataloading)
        
        dim_range = config.dim_range
        slice_win = config.slice_win
        strides = config.strides
        self.test_dates = config.test_dates
        if dataloading == "old":
            datamodule = LegacyDataLoading(self.cfg)
        else:

            datamodule = new_dataloading.FourDVarNetDataModule(
                slice_win=slice_win,
                dim_range=dim_range,
                strides=strides,
                **config.params['files_cfg'],
                **{k: tuple([slice(*dt) for dt in dts]) for k, dts in config.params['splits'].items()},
            )


        datamodule.setup()
        self.dataloaders = {
            'train': datamodule.train_dataloader(),
            'val': datamodule.val_dataloader(),
            'test': datamodule.val_dataloader(),
        }
        if dataloading == "old":
            self.var_Tr = datamodule.var_Tr
            self.var_Tt = datamodule.var_Tt
            self.var_Val = datamodule.var_Val
            self.mean_Tr = datamodule.mean_Tr
            self.mean_Tt = datamodule.mean_Tt
            self.mean_Val = datamodule.mean_Val
            self.min_lon, self.max_lon, self.min_lat, self.max_lat = -65, -55, 33, 43
            self.ds_size_time = 20
            self.ds_size_lon = 1
            self.ds_size_lat = 1
        else:
            self.mean_Tr = float(datamodule.norm_stats[0])
            self.mean_Tt = float(datamodule.norm_stats[0])
            self.mean_Val = float(datamodule.norm_stats[0])
            self.var_Tr = float(datamodule.norm_stats[1]) ** 2
            self.var_Tt = float(datamodule.norm_stats[1]) ** 2
            self.var_Val = float(datamodule.norm_stats[1]) ** 2
            self.min_lon, self.max_lon, self.min_lat, self.max_lat = (float(x) for x in datamodule.bounding_box)
            self.ds_size_time = float(datamodule.ds_size['time'])
            self.ds_size_lon = float(datamodule.ds_size['lon'])
            self.ds_size_lat = float(datamodule.ds_size['lat'])
        if self.cfg.stochastic == False:
            self.lit_cls = models.LitModelWithSST if dataloading == "with_sst" else models.LitModel
        else:
            self.lit_cls = LitModelStochastic

    def run(self, ckpt_path=None, dataloader="test", **trainer_kwargs):
        """
        Train and test model and run the test suite
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param dataloader: Dataloader on which to run the test Checkpoint from which to resume
        :param trainer_kwargs: (Optional)
        """
        mod, trainer = self.train(ckpt_path, **trainer_kwargs)
        self.test(dataloader=dataloader, _mod=mod, _trainer=trainer)

    def _get_model(self, ckpt_path=None):
        """
        Load model from ckpt_path or instantiate new model
        :param ckpt_path: (Optional) Checkpoint path to load
        :return: lightning module
        """
        # PENDING: do not pass norm stat if ckpt is provided
        if ckpt_path:
            mod = self.lit_cls.load_from_checkpoint(ckpt_path, ds_size_time=self.ds_size_time, test_dates=self.test_dates,
                                                    ds_size_lon=self.ds_size_lon,
                                                    ds_size_lat=self.ds_size_lat)
        else:
            mod = self.lit_cls(hparam=self.cfg, w_loss=self.wLoss,
                               mean_Tr=self.mean_Tr, mean_Tt=self.mean_Tt, mean_Val=self.mean_Val,
                               var_Tr=self.var_Tr, var_Tt=self.var_Tt, var_Val=self.var_Val,
                               min_lon=self.min_lon, max_lon=self.max_lon,
                               min_lat=self.min_lat, max_lat=self.max_lat,
                               ds_size_time=self.ds_size_time,
                               ds_size_lon=self.ds_size_lon,
                               ds_size_lat=self.ds_size_lat)
        return mod

    def train(self, ckpt_path=None, **trainer_kwargs):
        """
        Train a model
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param trainer_kwargs: (Optional) Trainer arguments
        :return:
        """

        mod = self._get_model(ckpt_path=ckpt_path)

        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename=self.filename_chkpt,
                                              save_top_k=3,
                                              mode='min')
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        num_gpus = torch.cuda.device_count()
        accelerator = "ddp" if (num_gpus * num_nodes) > 1 else None
        trainer = pl.Trainer(num_nodes=num_nodes, gpus=num_gpus, accelerator=accelerator, auto_select_gpus=True,
                             callbacks=[checkpoint_callback], **trainer_kwargs)
        trainer.fit(mod, self.dataloaders['train'], self.dataloaders['val'])
        return mod, trainer

    def test(self, ckpt_path=None, dataloader="test", _mod=None, _trainer=None, **trainer_kwargs):
        """
        Test a model
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param dataloader: Dataloader on which to run the test Checkpoint from which to resume
        :param trainer_kwargs: (Optional)
        """
        #mod = _mod or self._get_model(ckpt_path=ckpt_path)
        #ckpt_pth = '/gpfswork/rech/yrf/ueh53pd/4dvarnet-core/lightning_logs/version_296714/checkpoints/modelSLAInterpGF-Exp3-epoch=98-val_loss=0.05.ckpt'
        #ckpt_pth = '/gpfswork/rech/yrf/ueh53pd/4dvarnet-core/lightning_logs/version_297967/checkpoints/modelSLAInterpGF-Exp3-epoch=39-val_loss=0.12.ckpt'
        #ckpt_pth = '/gpfswork/rech/yrf/ueh53pd/4dvarnet-core/lightning_logs/version_318238/checkpoints/modelSLAInterpGF-Exp3-epoch=37-val_loss=0.12.ckpt'
        mod = self._get_model(ckpt_path=ckpt_path)
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        num_gpus = torch.cuda.device_count()
        accelerator = "ddp" if (num_gpus * num_nodes) > 1 else None
        # trainer = _trainer or pl.Trainer(num_nodes=num_nodes, gpus=num_gpus, accelerator=accelerator, **trainer_kwargs)
        trainer = pl.Trainer(num_nodes=1, gpus=1, accelerator=None, **trainer_kwargs)
        print(mod)
        trainer.test(mod, test_dataloaders=self.dataloaders[dataloader])

    def profile(self):
        """
        Run the profiling
        :return:
        """
        from pytorch_lightning.profiler import PyTorchProfiler

        profiler = PyTorchProfiler(
            "results/profile_report",
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=1),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_profile'),
            record_shapes=True,
            profile_memory=True,
        )
        self.train(
            **{
                'profiler': profiler,
                'max_epochs': 1,
            }
        )


if __name__ == '__main__':
    import fire

    fire.Fire(FourDVarNetRunner)
