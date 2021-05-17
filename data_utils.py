import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction import image
import solver as NN_4DVar


slice_args = {
        "time_min": "2013-08-23",
        "time_max": "2013-09-03",
        "lat_min": 30,
        "lat_max": 40,
        "lon_min": 295,
        "lon_max": 305,
    }


class SpaceTimePatches(Dataset):

    def __init__(self, path, slice_args):
        self.path       = path
        self.slice_args = slice_args

    def __getitem__(self, index):

        chunk = self.get_natl_slice(index)


    def __len__(self):

        pass


    def get_natl_slice(index):
        with xr.open_dataset(self.path) as ds:
            return (ds
                .pipe(lambda ds: ds.assign(time=ds.time.assign_attrs({'units': 'days since 2012-10-01'})))
                .pipe(xr.decode_cf)
                .pipe(lambda ds: ds.sel(time=slice(self.slice_args.get('time_min', "2012-10-01"),
                    self.slice_args.get('time_max', "2013-10-01"))))
                .pipe(lambda ds: ds.isel(y=(ds.lat > self.slice_args.get('lat_min', 0))))
                .pipe(lambda ds: ds.isel(y=(ds.lat < self.slice_args.get('lat_max', 360))))
                .pipe(lambda ds: ds.isel(x=(ds.lon < self.slice_args.get('lon_max', 360))))
                .pipe(lambda ds: ds.isel(x=(ds.lon > self.slice_args.get('lon_min', 0))))
            ).compute()



class Gradient_img(torch.nn.Module):
    def __init__(self, shapeData, shapeDate_test):
        super(Gradient_img, self).__init__()
        self.shapeData = shapeData
        self.shapeData_test = shapeDate_test

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))

    def forward(self, im, phase):
        if phase == 'test':
            shp_data = self.shapeData_test
        else:
            shp_data = self.shapeData

        if im.size(1) == 1:
            G_x = self.convGx(im)
            G_y = self.convGy(im)
            G = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2))
        else:
            for kk in range(0, im.size(1)):
                G_x = self.convGx(im[:, kk, :, :].view(-1, 1, shp_data[1], shp_data[2]))
                G_y = self.convGy(im[:, kk, :, :].view(-1, 1, shp_data[1], shp_data[2]))

                G_x = G_x.view(-1, 1, shp_data[1] - 2, shp_data[2] - 2)
                G_y = G_y.view(-1, 1, shp_data[1] - 2, shp_data[2] - 2)
                nG = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2))

                if kk == 0:
                    G = nG.view(-1, 1, shp_data[1] - 2, shp_data[2] - 2)
                else:
                    G = torch.cat((G, nG.view(-1, 1, shp_data[1] - 2, shp_data[2] - 2)), dim=1)
        return G


def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
    return patches.view(b, -1, patches.shape[-2], patches.shape[-1])


def extract_SpaceTimePatches(q, i1, i2, Wid, dT, rnd1, rnd2, D=1, Nbpatches=1, NoRndPatches=False):
    dataTraining = image.extract_patches_2d(
        np.moveaxis(q[i1:i2, ::D, ::D], 0, -1), (Wid, Wid), max_patches=Nbpatches, random_state=rnd1)
    dataTraining = np.moveaxis(dataTraining, -1, 1)
    dataTraining = dataTraining.reshape((Nbpatches, dataTraining.shape[1], Wid * Wid))

    if NoRndPatches == True:
        for ii in range(0, dataTraining.shape[1] - dT + 1):
            if ii == 0:
                temp = dataTraining[:, ii:ii + dT, :].reshape((1, dT, Nbpatches, Wid * Wid))
            else:
                temp = np.concatenate(
                    (temp, dataTraining[:, ii:ii + dT, :].reshape((1, dT, Nbpatches, Wid * Wid))), axis=0)

        dataTraining = np.moveaxis(temp, 1, 2)
    else:
        dataTraining = image.extract_patches_2d(dataTraining, (Nbpatches, dT),
                                                max_patches=dataTraining.shape[1] - dT + 1, random_state=rnd2)
        # dataTraining  = dataTraining.reshape((dT,W*W,Nbpatches*dataTraining.shape[-1]))
    dataTraining = dataTraining.reshape((dataTraining.shape[0], dataTraining.shape[1], dT, Wid, Wid))
    dataTraining = np.moveaxis(dataTraining, 0, -1)
    dataTraining = np.moveaxis(dataTraining, 0, -1)
    dataTraining = dataTraining.reshape((dT, Wid, Wid, dataTraining.shape[3] * dataTraining.shape[4]))
    dataTraining = np.moveaxis(dataTraining, -1, 0)
    return dataTraining


def save_NetCDF(saved_path1, x_test_rec):
    extent = [-65., -55., 30., 40.]
    indLat = 200
    indLon = 200

    lon = np.arange(extent[0], extent[1], 1 / (20 / dwscale))
    lat = np.arange(extent[2], extent[3], 1 / (20 / dwscale))
    indLat = int(indLat / dwscale)
    indLon = int(indLon / dwscale)
    lon = lon[:indLon]
    lat = lat[:indLat]

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


def compute_betas(train_loader, gradient_img, wLoss):
    # recompute the mean loss for OI when train_loader change
    running_loss_GOI = 0.
    running_loss_OI  = 0.
    num_loss         = 0.

    for targets_OI, targets_OI1, inputs_Mask, targets_GT in train_loader:
        # gradient norm field
        g_targets_GT = gradient_img(targets_GT, phase)
        loss_OI  = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, wLoss)
        loss_GOI = NN_4DVar.compute_WeightedLoss(gradient_img(targets_OI, phase) - g_targets_GT, wLoss)
        running_loss_GOI += loss_GOI.item() * targets_GT.size(0)
        running_loss_OI += loss_OI.item() * targets_GT.size(0)
        num_loss += targets_GT.size(0)

    epoch_loss_GOI = running_loss_GOI / num_loss
    epoch_loss_OI = running_loss_OI / num_loss

    betaX  = 1. / epoch_loss_OI
    betagX = 1. / epoch_loss_GOI

    print(".... MSE(Tr) OI %.3f -- MSE(Tr) gOI %.3f " % (epoch_loss_OI, epoch_loss_GOI))
    return betaX, betagX