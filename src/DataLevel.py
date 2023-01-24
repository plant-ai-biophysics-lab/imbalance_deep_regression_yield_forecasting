import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d




def return_num_samples_of_bins(df):

    dict_ = {}
    for i in range(30):
        if i == 0:
            Data  = df.loc[(df['ytrue'] >= i) & (df['ytrue'] <= (i+1))]
        elif i == 29:
            Data  = df.loc[(df['ytrue'] > i)]
        else: 
            Data  = df.loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))] 
        dict_[i+1] = len(Data) 

    return  dict_


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window 

def calc_lds_effective_dist(df, ks: int, sigma: int):  

    dict_ = return_num_samples_of_bins(df)
    emp_label_dist = [dict_.get(i, 0) for i in range(30)]
    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=10, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=ks, sigma=sigma)
    # calculate effective label distribution
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    return emp_label_dist, eff_label_dist