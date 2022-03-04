import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
import pandas as pd
from datetime import datetime, timedelta
from hydra.utils import get_class, instantiate, call
from omegaconf import OmegaConf
import hydra_config
import numpy as np

class FourDVarNetHydraRunner:
    def __init__(self, params, dm, lit_mod_cls, callbacks=None, logger=None):
        self.cfg = params
        self.filename_chkpt = self.cfg.ckpt_name
        self.callbacks = callbacks
        self.logger = logger
        self.dm = dm
        self.lit_cls = lit_mod_cls
        dm.setup()
        self.dataloaders = {
            'train': dm.train_dataloader(),
            'val': dm.val_dataloader(),
            'test': dm.test_dataloader(),
        }

        test_dates = np.concatenate([ \
                       [str(dt.date()) for dt in \
                       pd.date_range(dm.test_slices[i].start,dm.test_slices[i].stop)[(self.cfg.dT//2):-(self.cfg.dT//2)]] \
                      for i in range(len(dm.test_slices))])
        #print(test_dates)
        self.time = {'time_test' : test_dates}

        self.setup(dm)

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
            mod = self.lit_cls.load_from_checkpoint(ckpt_path,
                                                    w_loss=self.wLoss,
                                                    strict=False,
                                                    mean_Tr=self.mean_Tr,
                                                    mean_Tt=self.mean_Tt,
                                                    mean_Val=self.mean_Val,
                                                    var_Tr=self.var_Tr,
                                                    var_Tt=self.var_Tt,
                                                    var_Val=self.var_Val,
                                                    min_lon=self.min_lon, max_lon=self.max_lon,
                                                    min_lat=self.min_lat, max_lat=self.max_lat,
                                                    ds_size_time=self.ds_size_time,
                                                    ds_size_lon=self.ds_size_lon,
                                                    ds_size_lat=self.ds_size_lat,
                                                    time=self.time,
                                                    dX=self.dX, dY = self.dY,
                                                    swX=self.swX, swY=self.swY,
                                                    coord_ext={'lon_ext': self.lon,
                                                               'lat_ext': self.lat},
                                                    test_domain=self.cfg.test_domain,
                                                    resolution=self.resolution,
                                                    )

        else:
            mod = self.lit_cls(hparam=self.cfg,
                               w_loss=self.wLoss,
                               mean_Tr=self.mean_Tr,
                               mean_Tt=self.mean_Tt,
                               mean_Val=self.mean_Val,
                               var_Tr=self.var_Tr,
                               var_Tt=self.var_Tt,
                               var_Val=self.var_Val,
                               min_lon=self.min_lon, max_lon=self.max_lon,
                               min_lat=self.min_lat, max_lat=self.max_lat,
                               ds_size_time=self.ds_size_time,
                               ds_size_lon=self.ds_size_lon,
                               ds_size_lat=self.ds_size_lat,
                               time=self.time,
                               dX=self.dX, dY = self.dY,
                               swX=self.swX, swY=self.swY,
                               coord_ext = {'lon_ext': self.lon,
                                            'lat_ext': self.lat},
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

@hydra.main(config_path='hydra_config', config_name='main')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(seed=cfg.get('seed', None))
    dm = instantiate(cfg.datamodule)
    if cfg.get('callbacks') is not None:
        callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    else:
        callbacks=[]

    if cfg.get('logger') is not None:
        logger = instantiate(cfg.logger)
    else:
        logger=True
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls, callbacks=callbacks, logger=logger)
    call(cfg.entrypoint, self=runner)

if __name__ == '__main__':
    main()
