
import os
import numpy as np
import pandas as pd
import math
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from KDEpy import FFTKDE
from sklearn.preprocessing import MinMaxScaler





#=======================================================================================================#
#                                     LDS and inverse weight sampler                                    #
#=======================================================================================================#


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


def lds_prepare_weights(labels, reweight, max_target=30, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in range(max_target)}

    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: v for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    #print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        #print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

def cb_prepare_weights(labels, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, betha=2):
    value_dict = {x: 0 for x in range(30)}

    for label in labels:
        value_dict[min(30 - 1, int(label))] += 1

    value_dict = {k: v for k, v in value_dict.items()}  # clip weights for inverse re-weight


    emperical_values = [v for _, v in value_dict.items()]
    lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    #print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
    smoothed_value = convolve1d(
        np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
    num_per_label = [smoothed_value[min(30 - 1, int(label))] for label in labels]


    effective_value_norm = (num_per_label - np.min(num_per_label)) / (np.max(num_per_label) - np.min(num_per_label))
    weights = [np.float32((1-betha) / (1-(betha**x))) if (1-(betha**x)) != 0 else 1 for x in effective_value_norm]

    return weights



class check_lds_reweighting_():
    def __init__(self, df, rw_method: str, lds_ks: int, lds_sigma: int):
        self.df = df
        self.rw_method = rw_method
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma

    def return_pixelwise_weight(self):

        masks = None
        for idx, row in self.df.iterrows():
            xcoord     = row['X'] 
            ycoord     = row['Y'] 
            label_path = row['LABEL_PATH']
            mask  = self.crop_gen(label_path, xcoord, ycoord)
            mask  = np.swapaxes(mask, -1, 0)

            if masks is None: 
                masks = mask
            else: 
                masks = np.concatenate([masks, mask], axis = 0)


        labels_ = np.reshape(masks, (masks.shape[0]*masks.shape[1]*masks.shape[2]))

        weights, effective_value, emperical_values  = self.lds_prepare_weights(labels_*2.2417, max_target=70, lds=True)

        avg_weights = self.return_avg_weights_per_labels(np.array(weights), np.array(labels_))

        return weights, labels_, effective_value, emperical_values, avg_weights

    def return_avg_weights_per_labels(self, weights, labels):

        avg_weights = {}

        for i in range(70):
            if i == 0: 
                Ws = weights[np.where((labels >= i) &(labels <=(i+1)))]
                avg_weights[i] = np.mean(Ws)
            elif i == 69:
                Ws = weights[np.where((labels >= i))]
                avg_weights[i] = np.mean(Ws)
            else: 
                Ws = weights[np.where((labels >i) &(labels <= (i+1)))]
                avg_weights[i] = np.mean(Ws)

        avg_weights = list(avg_weights.values()) 
        avg_weights = [0 if math.isnan(x) else x for x in avg_weights]  

        return avg_weights


    def lds_prepare_weights(self, labels, max_target=30, lds=True):


        assert self.rw_method in {'none', 'inverse', 'sqrt_inv'}
        assert self.rw_method != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}

        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if self.rw_method == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif self.rw_method == 'inverse':
            value_dict = {k: v for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or self.rw_method == 'none':
            return None
        #print(f"Using re-weighting: [{reweight.upper()}]")
        emperical_vales = [v for _, v in value_dict.items()]

        if lds:
            lds_kernel_window = get_lds_kernel_window('gaussian', self.lds_ks, self.lds_sigma)
            #print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]

        return weights, smoothed_value, emperical_vales

    def crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        crop_src = src[:, xcoord:xcoord + 16, ycoord:ycoord + 16, :]
        return crop_src 

class check_cb_reweighting():
    def __init__(self, df, lds_ks: int, lds_sigma: int, betha: int):
        self.df = df
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        self.betha = betha

        
    def return_pixelwise_weight(self):
        masks = None
        for idx, row in self.df.iterrows():
            xcoord     = row['X'] 
            ycoord     = row['Y'] 
            label_path = row['LABEL_PATH'] 
            mask  = self.crop_gen(label_path, xcoord, ycoord) 
            mask  = np.swapaxes(mask, -1, 0)

            if masks is None: 
                masks = mask
            else: 
                masks = np.concatenate([masks, mask], axis = 0)


        labels_ = np.reshape(masks, (masks.shape[0]*masks.shape[1]*masks.shape[2]))


        weights, effective_value, emperical_values  = self.cb_prepare_weights(labels_, 
                                                                              lds_kernel='gaussian', 
                                                                              lds_ks=self.lds_ks, 
                                                                              lds_sigma=self.lds_sigma, 
                                                                              betha=self.betha)

        avg_weights = self.return_avg_weights_per_labels(np.array(weights), np.array(labels_))

        return weights, labels_, effective_value, emperical_values, avg_weights

    def cb_prepare_weights(self, labels, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, betha=2):
        value_dict = {x: 0 for x in range(30)}

        for label in labels:
            value_dict[min(30 - 1, int(label))] += 1

        value_dict = {k: v for k, v in value_dict.items()}  # clip weights for inverse re-weight


        emperical_values = [v for _, v in value_dict.items()]
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        #print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(30 - 1, int(label))] for label in labels]


        effective_value_norm = (num_per_label - np.min(num_per_label)) / (np.max(num_per_label) - np.min(num_per_label))
        weights = [np.float32((1-betha) / (1-(betha**x))) if (1-(betha**x)) != 0 else 50 for x in effective_value_norm]
        return weights, smoothed_value, emperical_values
    

    def return_avg_weights_per_labels(self, weights, labels):

        avg_weights = {}

        for i in range(30):
            if i == 0: 
                Ws = weights[np.where((labels >= i) &(labels <=(i+1)))]
                avg_weights[i] = np.mean(Ws)
            elif i == 29:
                Ws = weights[np.where((labels >= i))]
                avg_weights[i] = np.mean(Ws)
            else: 
                Ws = weights[np.where((labels >i) &(labels <= (i+1)))]
                avg_weights[i] = np.mean(Ws)

        avg_weights = list(avg_weights.values()) 
        avg_weights = [0 if math.isnan(x) else x for x in avg_weights]  

        return avg_weights


    def crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        crop_src = src[:, xcoord:xcoord + 16, ycoord:ycoord + 16, :]
        return crop_src 
    
class check_cost_sensitive_reweighting():
    def __init__(self, df):
        self.df = df

    def return_pixelwise_weight(self):
        pixel_values, pixel_stds, pixel_means, pixel_weights = self.calc_pixelwise_weight_std()

        avg_weights, avg_stds = self.return_avg_weights_per_bins(pixel_weights, pixel_stds, pixel_values)
        
        return avg_weights, avg_stds

    def calc_pixelwise_weight_std(self):
        pixel_values, pixel_stds, pixel_means, pixel_weights = [], [], [], []
        for idx, row in self.df.iterrows():
            xcoord     = row['X'] 
            ycoord     = row['Y'] 
            label_path = row['LABEL_PATH'] 
            label_mean = row['patch_mean'] 
            label_weight = row['NormWeight']


            mask  = self.crop_gen(label_path, xcoord, ycoord) 
            mask  = np.swapaxes(mask, -1, 0)
            mask_flat  = mask.flatten()
            arr_std    = abs(mask_flat - label_mean)

            l = len(mask_flat)
            pixel_values.append(mask_flat)
            pixel_stds.append(arr_std)
            pixel_means.append(l*[label_mean])
            pixel_weights.append(l*[label_weight])

        pixel_values = np.concatenate(pixel_values)
        pixel_stds = np.concatenate(pixel_stds)
        pixel_means = np.concatenate(pixel_means)
        pixel_weights = np.concatenate(pixel_weights)

        #print(f"{pixel_values.shape} |{pixel_stds.shape} |{pixel_means.shape} |{pixel_weights.shape} ")

        return pixel_values, pixel_stds, pixel_means, pixel_weights


    def return_avg_weights_per_bins(self, weights, stds, labels):

        avg_weights = {}
        avg_stds = {}

        for i in range(30):
            if i == 0: 
                Ws = weights[np.where((labels >= i) &(labels <=(i+1)))]
                avg_weights[i] = np.mean(Ws)
                std = stds[np.where((labels >= i) &(labels <=(i+1)))]
                avg_stds[i] = np.mean(std)
            elif i == 29:
                Ws = weights[np.where((labels >= i))]
                avg_weights[i] = np.mean(Ws)
                std = stds[np.where((labels >= i))]
                avg_stds[i] = np.mean(std)      
            else: 
                Ws = weights[np.where((labels >i) &(labels <= (i+1)))]
                avg_weights[i] = np.mean(Ws)
                std = stds[np.where((labels >i) &(labels <= (i+1)))]
                avg_stds[i] = np.mean(std) 

        avg_weights = list(avg_weights.values()) 
        avg_weights = [0 if math.isnan(x) else x for x in avg_weights]  
        avg_stds = list(avg_stds.values()) 

        return avg_weights, avg_stds


    def crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        crop_src = src[:, xcoord:xcoord + 16, ycoord:ycoord + 16, :]
        return crop_src 
    

#=======================================================================================================#
#                                             DenseWeight                                               #
#=======================================================================================================#
""" The code sources for this implementation modified based on the following link and GitHub repo for the paper 
titled "Density-based weighting for imbalanced regression". 


https://github.com/SteiMi/density-based-weighting-for-imbalanced-regression
"""

def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.
    From https://stackoverflow.com/a/41856629'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl

class TargetRelevance():

    def __init__(self, y, alpha=1.0):
        self.alpha = alpha
       #print('TargetRelevance alpha:', self.alpha)

        silverman_bandwidth = 1.06*np.std(y)*np.power(len(y), (-1.0/5.0))

        #print('Using Silverman Bandwidth', silverman_bandwidth)
        best_bandwidth = silverman_bandwidth

        self.kernel = FFTKDE(bw=best_bandwidth).fit(y, weights=None)

        x, y_dens_grid = self.kernel.evaluate(4096)  # Default precision is 1024
        self.x = x
        
        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens_grid = MinMaxScaler().fit_transform(y_dens_grid.reshape(-1, 1)).flatten()

        self.y_dens = np.vectorize(self.get_density)(y)

        self.eps = 1e-6
        w_star = np.maximum(1 - self.alpha * self.y_dens, self.eps)
        self.mean_w_star = np.mean(w_star)
        self.relevances = w_star / self.mean_w_star

    def get_density(self, y):
        idx = bisection(self.x, y)
        try:
            dens = self.y_dens_grid[idx]
        except IndexError:
            if idx <= -1:
                idx = 0
            elif idx >= len(self.x):
                idx = len(self.x) - 1
            dens = self.y_dens_grid[idx]
        return dens

    #@functools.lru_cache(maxsize=100000)
    def eval_single(self, y):
        dens = self.get_density(y)
        return np.maximum(1 - self.alpha * dens, self.eps) / self.mean_w_star

    def eval(self, y):
        ys = y.flatten().tolist()
        rels = np.array(list(map(self.eval_single, ys)))[:, None]
        return rels

    def __call__(self, y):
        return self.eval(y)#, self.get_density(y)
    

'''class TargetRelevance():

    def __init__(self, y, alpha: int):
        self.alpha = alpha
        self.y = y
        #print('TargetRelevance alpha:', self.alpha)
    def get_relevance(self):
        silverman_bandwidth = 1.06*np.std(self.y)*np.power(len(self.y), (-1.0/5.0))

        
        best_bandwidth = silverman_bandwidth
        #print(f'Using Silverman Bandwidth: {best_bandwidth}')

        self.kernel = FFTKDE(bw=best_bandwidth).fit(self.y, weights=None)

        self.y_min = self.y.min()
        self.y_max = self.y.max()

        x, y_dens = self.kernel.evaluate(self.y.shape[0])  # Default precision is 1024
        self.x = x

        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens = MinMaxScaler().fit_transform(y_dens.reshape(-1, 1)).flatten()

        self.eps = 1e-9
        w2 = np.maximum(1 - self.alpha * y_dens, self.eps)
        self.mean_w2 = np.mean(w2)
        relevances = w2 / self.mean_w2

        return relevances, self.y_dens,'''
#=======================================================================================================#
#                                     Emprical Weight Sampler                                           #
#=======================================================================================================#

def cost_sensitive_weight_sampler(df):
    
    Groups = df.groupby(by=["cultivar"])
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    
    dict_ = {}
    for state, frame in Groups:        
        count_list = frame['patch_mean'].value_counts(bins=bins, sort=False)
        count_sum = np.sum(count_list)
        
        dict_[state] = count_list, count_sum

    weight = []#np.zeros((len(df))) 
    
    for idx, row in df.iterrows():  
        patch_cultivar = row['cultivar']
        patch_mean = row['patch_mean']     

        get_patch_count = dict_[patch_cultivar][0][patch_mean]
        get_cultivar_sum = dict_[patch_cultivar][1]
        row_weight = get_patch_count / get_cultivar_sum
        weight.append(row_weight)
        
    weight = np.array(weight)
    df['weight'] = weight
    list_sum = df.groupby(by=["cultivar"])['weight'].transform('sum')
    NormWeights = df['weight']/list_sum
    df['NormWeight'] = NormWeights
    
    return df



def return_cost_sensitive_weight_sampler(train, val, test, save_dir, run_status: str):
    
    save_dir = os.path.join(save_dir, 'coords')

    if run_status == 'train':
        train_df   = cost_sensitive_weight_sampler(train)
        train_df.reset_index(inplace = True, drop = True)
        train_df.to_csv(os.path.join(save_dir,'train.csv'))
        
        valid_df  = cost_sensitive_weight_sampler(val)
        valid_df.reset_index(inplace = True, drop = True)
        valid_df.to_csv(os.path.join(save_dir,'val.csv'))
        
        
        test_df = cost_sensitive_weight_sampler(test)
        test_df.reset_index(inplace = True, drop = True)
        test_df.to_csv(os.path.join(save_dir, 'test.csv'))
    
    elif run_status == 'eval':
        train_df = pd.read_csv(os.path.join(save_dir,'train.csv'), index_col=0) 
        train_df.reset_index(inplace = True, drop = True)

        valid_df = pd.read_csv(os.path.join(save_dir,'val.csv'), index_col=0)
        valid_df.reset_index(inplace = True, drop = True)

        test_df = pd.read_csv(os.path.join(save_dir, 'test.csv'), index_col=0)
        test_df.reset_index(inplace = True, drop = True)

    train_weights = train_df['NormWeight'].to_numpy() 
    train_weights = torch.DoubleTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, 
                                                                   len(train_weights), replacement=True)    

    val_weights   = valid_df['NormWeight'].to_numpy() 
    val_weights   = torch.DoubleTensor(val_weights)
    val_sampler   = torch.utils.data.sampler.WeightedRandomSampler(val_weights, 
                                                                   len(val_weights), replacement=True)    

    test_weights = test_df['NormWeight'].to_numpy() 
    test_weights = torch.DoubleTensor(test_weights)
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights)) 
    
    
    return train_sampler, val_sampler, test_sampler
