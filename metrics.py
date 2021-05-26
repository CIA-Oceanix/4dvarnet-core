import datetime
import numpy as np
import xarray as xr


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