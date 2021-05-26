#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:59:23 2020
@author: rfablet
"""

import datetime

import numpy as np
import pytorch_lightning as pl
import torch
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import xarray as xr
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

import config
import solver as NN_4DVar
from models import Gradient_img, LitModel
from old_dataloading import LegacyDataLoading

cfg = OmegaConf.create(config.params)

datamodule = LegacyDataLoading(cfg)
datamodule.setup()
dataloaders = {
    'train': datamodule.train_dataloader(),
    'val': datamodule.val_dataloader(),
    'test': datamodule.val_dataloader(),
}
var_Tr = datamodule.var_Tr
var_Tt = datamodule.var_Tt
var_Val = datamodule.var_Val

#######################################Phi_r, Model_H, Model_Sampling architectures ################################################

print('........ Define AE architecture')
shapeData = cfg.shapeData  # np.array(x_train.shape[1:])
# shapeData_test = np.array(x_test.shape[1:])
# shapeData[0]  += shapeData[0]
# shapeData_test[0] += shapeData_test[0]

gradient_img = Gradient_img()

# model_LR = ModelLR()


# loss weghing wrt time
w_ = np.zeros(cfg.dT)
w_[int(cfg.dT / 2)] = 1.
wLoss = torch.Tensor(w_)

# recompute the MSE for OI on training dataset
# to define weighing parameters in the training

if cfg.betaX is None or cfg.betagX is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss_GOI = 0.
    running_loss_OI = 0.
    num_loss = 0

    gradient_img = gradient_img.to(device)
    wLoss = wLoss.to(device)

    for targets_OI, inputs_Mask, targets_GT in dataloaders['train']:
        targets_OI = targets_OI.to(device)
        inputs_Mask = inputs_Mask.to(device)
        targets_GT = targets_GT.to(device)

        # gradient norm field
        g_targets_GT = gradient_img(targets_GT)
        loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, wLoss)
        loss_GOI = NN_4DVar.compute_WeightedLoss(gradient_img(targets_OI) - g_targets_GT, wLoss)
        running_loss_GOI += loss_GOI.item() * targets_GT.size(0)
        running_loss_OI += loss_OI.item() * targets_GT.size(0)
        num_loss += targets_GT.size(0)

    epoch_loss_GOI = running_loss_GOI / num_loss
    epoch_loss_OI = running_loss_OI / num_loss

    betaX = 1. / epoch_loss_OI
    betagX = 1. / epoch_loss_GOI

    print(".... MSE(Tr) OI %.3f -- MSE(Tr) gOI %.3f " % (epoch_loss_OI, epoch_loss_GOI))
    print(".... betaX = %.3f -- betagX %.3f " % (betaX, betagX))


# print(f"{(betaX, betagX)=}")


def save_netcdf(saved_path1, x_test_rec, lon, lat):
    '''
    saved_path1: string 
    x_test_rec: 3d numpy array (4DVarNet-based predictions)
    lon: 1d numpy array 
    lat: 1d numpy array
    '''

    lon = np.arange(np.min(lon), np.max(lon) + 1. / (20. / dwscale), 1. / (20. / dwscale))
    lat = np.arange(np.min(lat), np.max(lat), 1. / (20. / dwscale))

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    indN_Tt = np.concatenate([np.arange(60, 80)])
    time = [datetime.datetime.strftime(datetime.datetime.strptime("2012-10-01", '%Y-%m-%d') \
                                       + datetime.timedelta(days=np.float64(i)), "%Y-%m-%d") for i in indN_Tt]

    xrdata = xr.Dataset( \
        data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                   'latitude': (('lat', 'lon'), mesh_lat), \
                   'Time': (('time'), time), \
                   'ssh': (('time', 'lat', 'lon'), x_test_rec[:, int(dT / 2), :, :])}, \
        coords={'lon': lon, 'lat': lat, 'time': indN_Tt})
    xrdata.time.attrs['units'] = 'days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=saved_path1, mode='w')


def nrmse(ref, pred):
    '''
    ref: Ground Truth fields
    pred: interpolated fields
    '''
    return np.sqrt(np.nanmean(((ref - np.nanmean(ref)) - (pred - np.nanmean(pred))) ** 2)) / np.nanstd(ref)


def nrmse_scores(gt, oi, pred, resfile):
    '''
    gt: 3d numpy array (Ground Truth)
    oi: 3d numpy array (OI)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    '''
    # Compute daily nRMSE scores
    nrmse_oi = []
    nrmse_pred = []
    for i in range(len(oi)):
        nrmse_oi.append(nrmse(gt[i], oi[i]))
        nrmse_pred.append(nrmse(gt[i], pred[i]))
    tab_scores = np.zeros((2, 3))
    tab_scores[0, 0] = np.nanmean(nrmse_oi)
    tab_scores[0, 1] = np.percentile(nrmse_oi, 5)
    tab_scores[0, 2] = np.percentile(nrmse_oi, 95)
    tab_scores[1, 0] = np.nanmean(nrmse_pred)
    tab_scores[1, 1] = np.percentile(nrmse_pred, 5)
    tab_scores[1, 2] = np.percentile(nrmse_pred, 95)
    np.savetxt(fname=resfile, X=tab_scores, fmt='%2.2f')
    return tab_scores


def compute_metrics(X_test, X_rec):
    # MSE
    mse = np.mean((X_test - X_rec) ** 2)

    # MSE for gradient
    gX_rec = np.gradient(X_rec, axis=[1, 2])
    gX_rec = np.sqrt(gX_rec[0] ** 2 + gX_rec[1] ** 2)

    gX_test = np.gradient(X_test, axis=[1, 2])
    gX_test = np.sqrt(gX_test[0] ** 2 + gX_test[1] ** 2)

    gmse = np.mean((gX_test - gX_rec) ** 2)
    ng = np.mean((gX_rec) ** 2)

    return {'mse': mse, 'mseGrad': gmse, 'meanGrad': ng}


w_loss = torch.nn.Parameter(torch.Tensor(w_), requires_grad=False)

if __name__ == '__main__':

    flagProcess = 0

    if flagProcess == 0:  ## training model from scratch

        loadTrainedModel = False  # True
        if loadTrainedModel == True:

            pathCheckPOint = "./SLANATL60_ChckPt/modelSLAInterpGF-Exp3-epoch=36-val_loss=0.05.ckpt"
            print('.... load pre-trained model :' + pathCheckPOint)
            mod = LitModel.load_from_checkpoint(pathCheckPOint)
            mod.hparams.n_grad = 10
            mod.hparams.iter_update = [0, 25, 40, 100, 150, 150, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update = [10, 10, 15, 15, 15, 20, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update = [1e-3, 1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        else:
            mod = LitModel(hparam=cfg, w_loss=w_loss, var_Val=var_Val, var_Tr=var_Tr, var_Tt=var_Tt)

        print(mod.hparams)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath=cfg.dir_save,
                                              filename='modelSLAInterpGF-Exp3-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='min')
        profiler_kwargs = {'max_epochs': 2}

        # trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        # trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1, **profiler_kwargs, callbacks=[checkpoint_callback])

        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])

        trainer.test(mod, test_dataloaders=dataloaders['val'])

        X_val = qHR[iiVal + int(dT / 2):jjVal - int(dT / 2), :, :]
        X_OI = qOI[iiVal + int(dT / 2):jjVal - int(dT / 2), :, :]

        val_mseRec = compute_metrics(X_val, mod.x_rec)
        val_mseOI = compute_metrics(X_val, X_OI)

        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        # ncfile = Dataset("results/test.nc","r")
        # X_rec  = ncfile.variables['ssh'][:]
        # ncfile.close()
        X_test = qHR[iiTest + int(dT / 2):jjTest - int(dT / 2), :, :]
        X_OI = qOI[iiTest + int(dT / 2):jjTest - int(dT / 2), :, :]

        test_mseRec = compute_metrics(X_test, mod.x_rec)
        test_mseOI = compute_metrics(X_test, X_OI)

        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%' % (
        val_mseOI['mse'], val_mseRec['mse'], 100. * (1. - val_mseRec['mse'] / val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%' % (
        val_mseOI['mseGrad'], val_mseRec['mseGrad'], 100. * (1. - val_mseRec['mseGrad'] / val_mseOI['meanGrad']),
        100. * (1. - val_mseRec['mseGrad'] / val_mseOI['meanGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%' % (
        test_mseOI['mse'], test_mseRec['mse'], 100. * (1. - test_mseRec['mse'] / test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%' % (
        test_mseOI['mseGrad'], test_mseRec['mseGrad'], 100. * (1. - test_mseRec['mseGrad'] / test_mseOI['meanGrad']),
        100. * (1. - test_mseRec['mseGrad'] / test_mseOI['meanGrad'])))

    elif flagProcess == 1:  ## profling

        mod = LitModel()
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
            record_shapes=True
        )
        # profile with max NbGradIter
        mod.NbGradIter[0] = mod.NbGradIter[-1]
        profiler_kwargs = {
            'profiler': profiler,
            'max_epochs': 1,
        }

        trainer = pl.Trainer(gpus=1, **profiler_kwargs)

        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])


    elif flagProcess == 2:  ## test trained model with the non-Lighning code
        mod = LitModel()
        fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_50_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_10_100_modelPHI_iter080.mod'

        mod.model.phi_r.load_state_dict(torch.load(fileAEModelInit))
        mod.model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter', '_modelGrad_iter')))
        mod.model.model_VarCost.load_state_dict(
            torch.load(fileAEModelInit.replace('_modelPHI_iter', '_modelVarCost_iter')))
        mod.model.n_grad = 10

        profiler_kwargs = {'max_epochs': 200}
        # trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1, **profiler_kwargs)
        trainer.test(mod, test_dataloaders=dataloaders['val'])

        # ncfile = Dataset("results/test.nc","r")
        # X_rec    = ncfile.variables['ssh'][:]
        # ncfile.close()
        X_val = qHR[iiVal + int(dT / 2):jjVal - int(dT / 2), :, :]
        X_OI = qOI[iiVal + int(dT / 2):jjVal - int(dT / 2), :, :]

        val_mseRec = compute_metrics(X_val, mod.x_rec)
        val_mseOI = compute_metrics(X_val, X_OI)

        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        # ncfile = Dataset("results/test.nc","r")
        # X_rec  = ncfile.variables['ssh'][:]
        # ncfile.close()
        X_test = qHR[iiTest + int(dT / 2):jjTest - int(dT / 2), :, :]
        X_OI = qOI[iiTest + int(dT / 2):jjTest - int(dT / 2), :, :]

        test_mseRec = compute_metrics(X_test, mod.x_rec)
        test_mseOI = compute_metrics(X_test, X_OI)

        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%' % (
        val_mseOI['mse'], val_mseRec['mse'], 100. * (1. - val_mseRec['mse'] / val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%' % (
        val_mseOI['mseGrad'], val_mseRec['mseGrad'], 100. * (1. - val_mseRec['mseGrad'] / val_mseOI['mseGrad']),
        100. * (1. - val_mseRec['mseGrad'] / val_mseOI['meanGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%' % (
        test_mseOI['mse'], test_mseRec['mse'], 100. * (1. - test_mseRec['mse'] / test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%' % (
        test_mseOI['mseGrad'], test_mseRec['mseGrad'], 100. * (1. - test_mseRec['mseGrad'] / test_mseOI['mseGrad']),
        100. * (1. - test_mseRec['mseGrad'] / test_mseOI['meanGrad'])))

    elif flagProcess == 3:  ## test trained model with the Lightning code

        pathCheckPOint = "./SLANATL60_ChckPt/modelSLAInterpGF-Exp3-epoch=36-val_loss=0.05.ckpt"

        mod = LitModel.load_from_checkpoint(pathCheckPOint)
        mod.model.n_grad = 10
        profiler_kwargs = {'max_epochs': 200}
        # trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1, **profiler_kwargs)
        trainer.test(mod, test_dataloaders=dataloaders['val'])

        X_val = qHR[iiVal + int(dT / 2):jjVal - int(dT / 2), :, :]
        X_OI = qOI[iiVal + int(dT / 2):jjVal - int(dT / 2), :, :]

        val_mseRec = compute_metrics(X_val, mod.x_rec)
        val_mseOI = compute_metrics(X_val, X_OI)

        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        X_test = qHR[iiTest + int(dT / 2):jjTest - int(dT / 2), :, :]
        X_OI = qOI[iiTest + int(dT / 2):jjTest - int(dT / 2), :, :]

        test_mseRec = compute_metrics(X_test, mod.x_rec)
        test_mseOI = compute_metrics(X_test, X_OI)

        saveRes = False
        if saveRes == True:
            save_NetCDF('results/test.nc', mod.x_rec)

        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%' % (
        val_mseOI['mse'], val_mseRec['mse'], 100. * (1. - val_mseRec['mse'] / val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%' % (
        val_mseOI['mseGrad'], val_mseRec['mseGrad'], 100. * (1. - val_mseRec['mseGrad'] / val_mseOI['meanGrad']),
        100. * (1. - val_mseRec['mseGrad'] / val_mseOI['meanGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%' % (
        test_mseOI['mse'], test_mseRec['mse'], 100. * (1. - test_mseRec['mse'] / test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%' % (
        test_mseOI['mseGrad'], test_mseRec['mseGrad'], 100. * (1. - test_mseRec['mseGrad'] / test_mseOI['meanGrad']),
        100. * (1. - test_mseRec['mseGrad'] / test_mseOI['meanGrad'])))
