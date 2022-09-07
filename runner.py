import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import PyTorchProfiler

import os
import pandas as pd
import numpy as np

def get_profiler():
    print( torch.profiler.ProfilerActivity.CPU,)
    return PyTorchProfiler(
            "results/profile_report",
            schedule=torch.profiler.schedule(
                skip_first=2,
                wait=2,
                warmup=2,
                active=2),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_profile'),
            record_shapes=True,
            profile_memory=True,
    )


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
                                                    hparam=self.cfg,
                                                    strict=False,
                                                    test_domain=self.cfg.test_domain,
                                                    mean_Tr=self.mean_Tr,
                                                    mean_Tt=self.mean_Tt,
                                                    mean_Val=self.mean_Val,
                                                    var_Tr=self.var_Tr,
                                                    var_Tt=self.var_Tt,
                                                    var_Val=self.var_Val,
                                                    )

        else:
            mod = self.lit_cls(hparam=self.cfg,
                               mean_Tr=self.mean_Tr,
                               mean_Tt=self.mean_Tt,
                               mean_Val=self.mean_Val,
                               var_Tr=self.var_Tr,
                               var_Tt=self.var_Tt,
                               var_Val=self.var_Val,
                               test_domain=self.cfg.test_domain,
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
        gpus = trainer_kwargs.get('gpus', torch.cuda.device_count())

        num_gpus = gpus if isinstance(gpus, (int, float)) else  len(gpus) if hasattr(gpus, '__len__') else 0
        accelerator = "ddp" if (num_gpus * num_nodes) > 1 else None
        trainer_kwargs_final = {**dict(num_nodes=num_nodes, gpus=gpus, logger=self.logger, strategy=accelerator, auto_select_gpus=(num_gpus * num_nodes) > 0,
                             callbacks=[checkpoint_callback, lr_monitor]),  **trainer_kwargs}
        print(trainer_kwargs)
        print(trainer_kwargs_final)
        trainer = pl.Trainer(**trainer_kwargs_final)
        trainer.fit(mod, self.dataloaders['train'], self.dataloaders['val'])
        return mod, trainer

    def test(self, ckpt_path=None, dataloader="test", _mod=None, _trainer=None, **trainer_kwargs):
        """
        Test a model
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param dataloader: Dataloader on which to run the test Checkpoint from which to resume
        :param trainer_kwargs: (Optional)
        """

        if _trainer is not None:
            _trainer.test(mod, dataloaders=self.dataloaders[dataloader])
            return

        mod = _mod or self._get_model(ckpt_path=ckpt_path)

        trainer_kwargs_final = {**dict(num_nodes=1, gpus=1, accelerator=None), **trainer_kwargs}
        trainer = pl.Trainer(**trainer_kwargs_final)
        trainer.test(mod, dataloaders=self.dataloaders[dataloader])
        return mod

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
