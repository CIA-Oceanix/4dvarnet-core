import numpy as np
import scipy.ndimage as nd

def imputing_nan(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell
    """
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)

def hanning2d(M, N):
    """
    A 2D hanning window, as per IDL's hanning function.  See numpy.hanning for the 1d description
    """

    if N <= 1:
        return np.hanning(M)
    elif M <= 1:
        return np.hanning(N) # scalar unity; don't window if dims are too small
    else:
        return np.outer(np.hanning(M),np.hanning(N))

def avg_rapsd2dv1(img3d,res,hanning):
    """ Computes and plots radially averaged power spectral density mean (power
     spectrum) of an image set img3d along the first dimension.
    """
    N = img3d.shape[0]
    for i in range(N):
        img=img3d[i,:,:]
        f_, Pf_ = rapsd2dv1(img,res,hanning)
        if i==0:
            f, Pf = f_, Pf_
        else:
            f = np.vstack((f,f_))
            Pf= np.vstack((Pf,Pf_))
    Pf = np.mean(Pf,axis=0)
    return f_, Pf

def avg_err_rapsd2dv1(img3d,img3dref,res,hanning):
    """ Computes and plots radially averaged power spectral density error mean (power
     spectrum) of an image set img3d along the first dimension.
    """
    n = img3d.shape[0]
    for i in range(n):
        img1 = img3d[i,:,:]
        img2 = img3dref[i,:,:]
        f_, pf_ = rapsd2dv1(img1-img2,res,hanning)
        pf_ = (pf_/rapsd2dv1(img2,res,hanning)[1])
        if i==0:
            f, pf = f_, pf_
        else:
            f = np.vstack((f,f_))
            pf= np.vstack((pf,pf_))
    pf = np.mean(pf,axis=0)
    return f_, pf


def err_rapsd2dv1(img,imgref,res,hanning):
    """ Computes and plots radially averaged power spectral density error (power
     spectrum).
    """
    f_, pf_ = rapsd2dv1(img-imgref,res,hanning)
    pf_     = (pf_/rapsd2dv1(imgref,res,hanning)[1])
    return f_, pf_

def rapsd2dv1(img,res,hanning):
    """ Computes and plots radially averaged power spectral density (power
     spectrum) of image IMG with spatial resolution RES.
    """
    img = img.copy()
    n, m = img.shape
    if hanning:
        img = hanning2d(*img.shape) * img
    img = imputing_nan(img)
    imgf = np.fft.fftshift(np.fft.fft2(img))
    imgfp = np.power(np.abs(imgf)/(n*m),2)
    # Adjust PSD size
    dim_diff = np.abs(n-m)
    dim_max = max(n,m)
    if (n>m):
        if ((dim_diff%2)==0):
            imgfp = np.pad(imgfp,((0,0),(int(dim_diff/2),int(dim_diff/2))),'constant',constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp,((0,0),(int(dim_diff/2),1+int(dim_diff/2))),'constant',constant_values=np.nan)

    elif (n<m):
        if ((dim_diff%2)==0):
            imgfp = np.pad(imgfp,((int(dim_diff/2),int(dim_diff/2)),(0,0)),'constant',constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp,((int(dim_diff/2),1+int(dim_diff/2)),(0,0)),'constant',constant_values=np.nan)
    half_dim = int(np.ceil(dim_max/2.))
    x, y = np.meshgrid(np.arange(-dim_max/2.,dim_max/2.-1+0.00001),np.arange(-dim_max/2.,dim_max/2.-1+0.00001))
    theta, rho = cart2pol(x, y)
    rho = np.round(rho+0.5)
    pf = np.zeros(half_dim)
    f1 = np.zeros(half_dim)
    for r in range(half_dim):
      pf[r] = np.nansum(imgfp[rho == (r+1)])
      f1[r] = float(r+1)/dim_max
    f1 = f1/res
    return f1, pf



