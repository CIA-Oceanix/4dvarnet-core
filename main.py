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


class FourDVarNetRunner:
    def __init__(self, dataloading="old", config=None):
        from lit_model import LitModel
        from lit_model_sst import LitModelWithSST
        from lit_model_stochastic import LitModelStochastic
        from new_dataloading import FourDVarNetDataModule
        from old_dataloading import LegacyDataLoading
        self.filename_chkpt = 'modelSLAInterpGF-Exp3-{epoch:02d}-{val_loss:.2f}'
        if config is None:
            import config
        else:
            import importlib
            print('loading config')
            config = importlib.import_module("config_" + str(config))

        self.cfg = OmegaConf.create(config.params)
        print(OmegaConf.to_yaml(self.cfg))
        dataloading = config.params['dataloading']

        dim_range = config.dim_range
        slice_win = config.slice_win
        strides = config.strides
        time_period = config.time_period

        if dataloading == "old":
            datamodule = LegacyDataLoading(self.cfg)
        else:
            datamodule = FourDVarNetDataModule(
                slice_win=slice_win,
                dim_range=dim_range,
                strides=strides,
                train_slices=config.time_period['train_slices'],
                test_slices=config.time_period['test_slices'],
                val_slices=config.time_period['val_slices'],
                resize_factor=config.params['resize_factor'],
                **config.params['files_cfg']
            )

        datamodule.setup()
        self.dataloaders = {
            'train': datamodule.train_dataloader(),
            'val': datamodule.val_dataloader(),
            'test': datamodule.test_dataloader(),
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
            self.setup(datamodule=datamodule)

        if config.params['stochastic'] == False:
            self.lit_cls = LitModelWithSST if dataloading == "with_sst" else LitModel
        else:
            self.lit_cls = LitModelStochastic

        self.time = config.time

    def setup(self, datamodule):
        self.mean_Tr = datamodule.norm_stats[0]
        self.mean_Tt = datamodule.norm_stats[0]
        self.mean_Val = datamodule.norm_stats[0]
        self.var_Tr = datamodule.norm_stats[1] ** 2
        self.var_Tt = datamodule.norm_stats[1] ** 2
        self.var_Val = datamodule.norm_stats[1] ** 2
        self.min_lon = datamodule.dim_range['lon'].start
        self.max_lon = datamodule.dim_range['lon'].stop
        self.min_lat = datamodule.dim_range['lat'].start
        self.max_lat = datamodule.dim_range['lat'].stop
        self.ds_size_time = datamodule.ds_size['time']
        self.ds_size_lon = datamodule.ds_size['lon']
        self.ds_size_lat = datamodule.ds_size['lat']
        self.dX = int((datamodule.slice_win['lon']-datamodule.strides['lon'])/2)
        self.dY = int((datamodule.slice_win['lat']-datamodule.strides['lat'])/2)
        self.swX = datamodule.slice_win['lon']
        self.swY = datamodule.slice_win['lat']
        self.lon, self.lat = datamodule.coordXY()
        w_ = np.zeros(self.cfg.dT)
        w_[int(self.cfg.dT / 2)] = 1.
        self.wLoss = torch.Tensor(w_)
        self.resolution = datamodule.resolution

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
        print('get_model: ', ckpt_path)
        if ckpt_path:
            mod = self.lit_cls.load_from_checkpoint(ckpt_path, w_loss=self.wLoss, strict=False,
                                                    mean_Tr=self.mean_Tr, mean_Tt=self.mean_Tt, mean_Val=self.mean_Val,
                                                    var_Tr=self.var_Tr, var_Tt=self.var_Tt, var_Val=self.var_Val,
                                                    min_lon=self.min_lon, max_lon=self.max_lon,
                                                    min_lat=self.min_lat, max_lat=self.max_lat,
                                                    ds_size_time=self.ds_size_time,
                                                    ds_size_lon=self.ds_size_lon,
                                                    ds_size_lat=self.ds_size_lat,
                                                    time=self.time,
                                                    dX = self.dX, dY = self.dY,
                                                    swX = self.swX, swY = self.swY,
                                                    coord_ext = {'lon_ext': self.lon, 'lat_ext': self.lat},
                                                    resolution=self.resolution,
                                                    )

        else:
            mod = self.lit_cls(hparam=self.cfg, w_loss=self.wLoss,
                               mean_Tr=self.mean_Tr, mean_Tt=self.mean_Tt, mean_Val=self.mean_Val,
                               var_Tr=self.var_Tr, var_Tt=self.var_Tt, var_Val=self.var_Val,
                               min_lon=self.min_lon, max_lon=self.max_lon,
                               min_lat=self.min_lat, max_lat=self.max_lat,
                               ds_size_time=self.ds_size_time,
                               ds_size_lon=self.ds_size_lon,
                               ds_size_lat=self.ds_size_lat,
                               time=self.time,
                               dX = self.dX, dY = self.dY,
                               swX = self.swX, swY = self.swY,
                               coord_ext = {'lon_ext': self.lon, 'lat_ext': self.lat},
                               resolution=self.resolution,
                               )
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
        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        num_gpus = torch.cuda.device_count()
        accelerator = "ddp" if (num_gpus * num_nodes) > 1 else None
        trainer = pl.Trainer(num_nodes=num_nodes, gpus=num_gpus, accelerator=accelerator, auto_select_gpus=(num_gpus * num_nodes) > 0,
                             callbacks=[checkpoint_callback, lr_monitor], **trainer_kwargs)
        trainer.fit(mod, self.dataloaders['train'], self.dataloaders['val'])
        return mod, trainer

    def test(self, ckpt_path=None, dataloader="test", _mod=None, _trainer=None, **trainer_kwargs):
        """
        Test a model
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param dataloader: Dataloader on which to run the test Checkpoint from which to resume
        :param trainer_kwargs: (Optional)
        """

        mod = _mod or self._get_model(ckpt_path=ckpt_path)

        trainer = pl.Trainer(num_nodes=1, gpus=1, accelerator=None, **trainer_kwargs)
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
