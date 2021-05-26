import pytorch_lightning as pl
from torch.utils.data import DataLoader
from netCDF4 import Dataset
import torch
import numpy as np
import os
from sklearn.feature_extraction import image


class LegacyDataLoading(pl.LightningDataModule):
    """
    This is the old way of loading data,
    this file is to be deleted as soon as we tested that the new dataloading utilities are OK
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.var_Tr = None
        self.var_Tt = None
        self.var_Val = None
        self.training_dataset=None
        self.val_dataset=None
        self.test_dataset=None


    def setup(self, stage=None):
        # Definiton of training, validation and test dataset
        # from dayly indices over a one-year time series
        cfg = self.cfg
        iiTr1 = 0
        jjTr1 = 50 - int(cfg.dT / 2)

        iiTr2 = 130 + int(cfg.dT / 2)
        jjTr2 = 365

        iiVal = 60 - int(cfg.dT / 2)
        jjVal = 80 + int(cfg.dT / 2)

        iiTest = 90 - int(cfg.dT / 2)
        jjTest = 110 + int(cfg.dT / 2)

        ############################################## Data generation ###############################################################
        print('........ Random seed set to 100')
        np.random.seed(100)
        torch.manual_seed(100)

        ncfile = Dataset(os.path.join(cfg.data_dir, 'NATL60-CJM165_NATL_ssh_y2013.1y.nc'), "r")

        # select GF region
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        idLat = np.where((lat >= 33.) & (lat <= 43.))[0]
        idLon = np.where((lon >= -65.) & (lon <= -55.))[0]
        lon = lon[idLon]
        lat = lat[idLat]

        # dirSAVE = '/gpfswork/rech/yrf/uba22to/ResSLANATL60/'
        os.makedirs(cfg.dir_save, exist_ok=True)

        genSuffixObs = ''
        ncfile = Dataset(os.path.join(cfg.data_dir, 'NATL60-CJM165_NATL_ssh_y2013.1y.nc'), "r")
        qHR = ncfile.variables['ssh'][:, idLat, idLon]
        ncfile.close()

        if cfg.flagSWOTData == True:
            print('.... Use SWOT+4-nadir dataset')
            genFilename = 'resInterpSWOTSLAwOInoSST_' + str('%03d' % (cfg.W)) + 'x' + str('%03d' % (cfg.W)) + 'x' + str(
                '%02d' % (cfg.dT))
            # OI data using a noise-free OSSE (ssh_mod variable)
            ncfile = Dataset(os.path.join(cfg.data_dir, 'ssh_NATL60_swot_4nadir.nc'), "r")
            qOI = ncfile.variables['ssh_mod'][:, idLat, idLon]
            ncfile.close()

            # OI data using a noise-free OSSE (ssh_mod variable)
            ncfile = Dataset(os.path.join(cfg.data_dir, 'dataset_nadir_0d_swot.nc'), "r")
            qMask = ncfile.variables['ssh_mod'][:, idLat, idLon]
            qMask = 1.0 - qMask.mask.astype(float)
            ncfile.close()
        else:
            print('.... Use SWOT+4-nadir dataset')
            genFilename = 'resInterpSWOTSLAwOInoSST_' + str('%03d' % (cfg.W)) + 'x' + str('%03d' % (cfg.W)) + 'x' + str(
                '%02d' % (cfg.dT))
            # OI data using a noise-free OSSE (ssh_mod variable)
            ncfile = Dataset(os.path.join(cfg.data_dir, 'ssh_NATL60_4nadir.nc'), "r")
            qOI = ncfile.variables['ssh_mod'][:, idLat, idLon]
            ncfile.close()

            # OI data using a noise-free OSSE (ssh_mod variable)
            ncfile = Dataset(os.path.join(cfg.data_dir, 'dataset_nadir_0d.nc'), "r")
            qMask = ncfile.variables['ssh_mod'][:, idLat, idLon]
            qMask = 1.0 - qMask.mask.astype(float)
            ncfile.close()

        print('----- MSE OI: %.3f' % np.mean((qOI - qHR) ** 2))
        print()

        ## extraction of patches from the SSH field
        # NoRndPatches = False
        # if ( Nbpatches == 1 ) & ( W == 200 ):
        # NoRndPatches = True
        print('... No random seed for the extraction of patches')

        qHR = qHR[:, 0:200, 0:200]
        qOI = qOI[:, 0:200, 0:200]
        qMask = qMask[:, 0:200, 0:200]

        def extract_SpaceTimePatches(q, i1, i2, W, dT, rnd1, rnd2, Nbpatches, D=1):
            dataTraining = image.extract_patches_2d(np.moveaxis(q[i1:i2, ::D, ::D], 0, -1), (W, W),
                                                    max_patches=Nbpatches, random_state=rnd1)
            dataTraining = np.moveaxis(dataTraining, -1, 1)
            dataTraining = dataTraining.reshape((Nbpatches, dataTraining.shape[1], W * W))

            dataTraining = image.extract_patches_2d(dataTraining, (Nbpatches, dT), max_patches=None)

            dataTraining = dataTraining.reshape((dataTraining.shape[0], dataTraining.shape[1], dT, W, W))
            dataTraining = np.moveaxis(dataTraining, 0, -1)
            dataTraining = np.moveaxis(dataTraining, 0, -1)
            dataTraining = dataTraining.reshape((dT, W, W, dataTraining.shape[3] * dataTraining.shape[4]))
            dataTraining = np.moveaxis(dataTraining, -1, 0)
            return dataTraining

            # training dataset

        dataTraining1 = extract_SpaceTimePatches(qHR, iiTr1, jjTr1, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                                 cfg.dx)
        dataTrainingMask1 = extract_SpaceTimePatches(qMask, iiTr1, jjTr1, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2,
                                                     cfg.Nbpatches, cfg.dx)
        dataTrainingOI1 = extract_SpaceTimePatches(qOI, iiTr1, jjTr1, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                                   cfg.dx)

        dataTraining2 = extract_SpaceTimePatches(qHR, iiTr2, jjTr2, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                                 cfg.dx)
        dataTrainingMask2 = extract_SpaceTimePatches(qMask, iiTr2, jjTr2, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2,
                                                     cfg.Nbpatches, cfg.dx)
        dataTrainingOI2 = extract_SpaceTimePatches(qOI, iiTr2, jjTr2, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                                   cfg.dx)

        dataTraining = np.concatenate((dataTraining1, dataTraining2), axis=0)
        dataTrainingMask = np.concatenate((dataTrainingMask1, dataTrainingMask2), axis=0)
        dataTrainingOI = np.concatenate((dataTrainingOI1, dataTrainingOI2), axis=0)

        # test dataset
        dataTest = extract_SpaceTimePatches(qHR, iiTest, jjTest, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                            cfg.dx)
        dataTestMask = extract_SpaceTimePatches(qMask, iiTest, jjTest, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                                cfg.dx)
        dataTestOI = extract_SpaceTimePatches(qOI, iiTest, jjTest, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                              cfg.dx)

        # validation dataset
        dataVal = extract_SpaceTimePatches(qHR, iiVal, jjVal, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)
        dataValMask = extract_SpaceTimePatches(qMask, iiVal, jjVal, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                               cfg.dx)
        dataValOI = extract_SpaceTimePatches(qOI, iiVal, jjVal, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,
                                             cfg.dx)

        meanTr = np.mean(dataTraining)
        x_train = dataTraining - meanTr
        stdTr = np.sqrt(np.mean(x_train ** 2))
        x_train = x_train / stdTr

        x_trainOI = (dataTrainingOI - meanTr) / stdTr
        x_trainMask = dataTrainingMask

        x_test = (dataTest - meanTr)
        stdTt = np.sqrt(np.mean(x_test ** 2))
        x_test = x_test / stdTr
        x_testOI = (dataTestOI - meanTr) / stdTr
        x_testMask = dataTestMask

        x_val = (dataVal - meanTr)
        stdVal = np.sqrt(np.mean(x_val ** 2))
        x_val = x_val / stdTr
        x_valOI = (dataValOI - meanTr) / stdTr
        x_valMask = dataValMask

        print('----- MSE Tr OI: %.6f' % np.mean(
            (dataTrainingOI[:, int(cfg.dT / 2), :, :] - dataTraining[:, int(cfg.dT / 2), :, :]) ** 2))
        print('----- MSE Tt OI: %.6f' % np.mean(
            (dataTestOI[:, int(cfg.dT / 2), :, :] - dataTest[:, int(cfg.dT / 2), :, :]) ** 2))

        print('..... Training dataset: %dx%dx%dx%d' % (
        x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        print('..... Test dataset    : %dx%dx%dx%d' % (
        x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]))

        print('..... Masked points (Tr)) : %.3f' % (np.sum(x_trainMask) / (
                    x_trainMask.shape[0] * x_trainMask.shape[1] * x_trainMask.shape[2] * x_trainMask.shape[3])))
        print('..... Masked points (Tt)) : %.3f' % (np.sum(x_testMask) / (
                    x_testMask.shape[0] * x_testMask.shape[1] * x_testMask.shape[2] * x_testMask.shape[3])))

        print('----- MSE Tr OI: %.6f' % np.mean(
            stdTr ** 2 * (x_trainOI[:, int(cfg.dT / 2), :, :] - x_train[:, int(cfg.dT / 2), :, :]) ** 2))
        print('----- MSE Tt OI: %.6f' % np.mean(
            stdTr ** 2 * (x_testOI[:, int(cfg.dT / 2), :, :] - x_test[:, int(cfg.dT / 2), :, :]) ** 2))

        ######################### data loaders
        self.training_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI), torch.Tensor(x_trainMask),
                                                          torch.Tensor(x_train))  # create your datset
        self.val_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_valOI), torch.Tensor(x_valMask),
                                                     torch.Tensor(x_val))  # create your datset
        self.test_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_testOI), torch.Tensor(x_testMask),
                                                      torch.Tensor(x_test))  # create your datset

        self.var_Tr = np.var(x_train)

        self.var_Tt = np.var(x_test)

        self.var_Val = np.var(x_val)


    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                    num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)

