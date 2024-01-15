import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set(font_scale=1.5)
sns.set_theme(style='white')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr

import sys
sys.path.append('../')
from models import configs
from utils import utils, RWSampler

#=============================================================================================#
#=============================================================================================#
#=============================================================================================#
EXTREME_LOWER_THRESHOLD = 22.24
EXTREME_UPPER_THRESHOLD = 54.36
HECTARE_TO_ACRE_SCALE = 2.471  # 2.2417
MAXIMUM_AXIS_VALUE = 75
WEEK_FOR_VIS = 15

PLOT_CMAP = 'viridis'
PLOT_MINCNT = 100
PLOT_VIS_FACE_COLOR = 'white'
PLOT_BOX_FACE_COLOR = 'grey'
PLOT_TEXT_FONTSIZE = 13

#=========================================================================#
#================================ Helpers ================================#
#=========================================================================#
def regression_metrics(ytrue, ypred):
    """Calculating the evaluation metric based on regression results:
    input:
            ytrue: 
            ypred:
            
    output:
            root mean square error(rmse)
            root square(r^2)
            mean absolute error(mae)
            mean absolute percent error(mape)
            mean of true yield
            mean of prediction yield
    """

    r_square   = r2_score(ytrue, ypred)
    mae        = mean_absolute_error(ytrue, ypred)
    rmse       = np.sqrt(mean_squared_error(ytrue, ypred))
    mape       = mean_absolute_percentage_error(ytrue, ypred)*100
    mean_ytrue = np.mean(ytrue)
    mean_ypred = np.mean(ypred)
    
    return [r_square , mae, rmse, mape, mean_ytrue, mean_ypred] 

def apply_percentile(row, target_percentile):
    # Check if row is array-like, if not, make it an array
    if not isinstance(row, np.ndarray):
        row = np.array([row])
    return np.percentile(row, target_percentile)

def safe_mean(x):
    if isinstance(x, np.ndarray):
        return np.mean(x)
    else:
        return x
    
def select_closest_percentile(target_key, percentile_dict):

    closest_key = min(percentile_dict.keys(), key=lambda x: abs(x - target_key))
    return percentile_dict[closest_key]

def find_percentile_of_target(target, values_list):
    """
    Find the percentile of a target value within a list of values.

    Args:
    target (float or int): The target value.
    values_list (list): A list of numeric values.

    Returns:
    float: The percentile of the target within the list.
    """
    sorted_values = sorted(values_list)

    less_than_target = sum(value <= target for value in sorted_values)

    percentile = 100 * less_than_target / len(values_list)

    return percentile

def _find_closest_value(my_list, target):
    closest_value = my_list[0]
    minimum_diff = abs(my_list[0] - target)
    for value in my_list:
        diff = abs(value - target)
        if diff < minimum_diff:
            closest_value = value
            minimum_diff = diff
    return closest_value

def _return_shorter_df(df, category: str):

    blocks = df.groupby(by='block')
    names, ytrue, ypred = [], [], []
    cultivar, x, y = [], [], []
    closests, percentiles = [], []

    for b, bdf in blocks:
        same_coords = bdf.groupby(['x', 'y'])
        
        for g, c in same_coords: 
            if c.shape[0] > 1:
                mean_ytrue = c['ytrue'].mean().astype('float32')
                list_pred = c['ypred_w15'].values.astype('float32')
                closest    = _find_closest_value(c['ypred_w15'].values, mean_ytrue)
                percentile = find_percentile_of_target(closest, c['ypred_w15'].values)
            else: 
                mean_ytrue = c['ytrue'].values[0].astype('float32')
                list_pred = c['ypred_w15'].values[0].astype('float32')
                closest    = list_pred
                percentile = 50


            names.append(b)
            cultivar.append(c.iloc[0]['cultivar'])
            x.append(c.iloc[0]['x'])
            y.append(c.iloc[0]['y'])
            ytrue.append(mean_ytrue)
            ypred.append(list_pred)
            closests.append(closest)
            percentiles.append(percentile)

    # Create DataFrame
    m_df = pd.DataFrame({
        'block': names,
        'cultivar': cultivar,
        'x': x, 
        'y': y,
        'ytrue': ytrue,
        'ypred_w15': ypred,
        'closest': closests,
        'p': percentiles,
    })

    if category == 'train':
        m_df['ytrue_rounded'] = np.floor(m_df['ytrue'])
        mean_percentile_per_bin = m_df.groupby('ytrue_rounded')['p'].mean()
        # Convert the Series to a dictionary
        mean_percentile_dict = mean_percentile_per_bin.to_dict()

        return mean_percentile_dict
    else:
        return m_df
    
def return_iter_test_dfs(dir):
    list_df = []
    for i in range(10):
        test_df = pd.read_csv(os.path.join(dir + f'_test_{i}.csv'), index_col=0)
        # test_df = return_modified_df(test_df)
        list_df.append(test_df)

    return list_df

def aggregate(src, scale):
    
    w = int(src.shape[0]/scale)
    h = int(src.shape[1]/scale)
    mtx = np.full((w, h), -1, dtype=np.float32)
    for i in range(w):
        for j in range(h):   
            mtx[i,j]=np.mean(src[i*scale:(i+1)*scale, j*scale:(j+1)*scale])                    
    return mtx        

def return_modified_df(test_df, cat: str):

    # short_test_df = _return_shorter_df(test_df, category= 'test')
    # mean_percentile_dict = _return_shorter_df(train_df, category= 'train')
    
    results = {
        'block': [], 'x': [], 'y': [], 'cultivar': [],
        'ytrue': [],
        **{f'ypred_w{i}': [] for i in range(1, 16)}
    }

    
    def process_group(name, group):
        mean_ytrue = group['ytrue'].mean().astype('float32')

        if mean_ytrue < 9:
            if cat == 'extreme': 
                percentile_value = 5
            elif cat == 'mean': 
                percentile_value = 50
        elif mean_ytrue >= 20:
            if cat == 'extreme': 
                percentile_value = 95
            elif cat == 'mean': 
                percentile_value = 50
        else:
            percentile_value = None 

        for i in range(1, 16):
            key = f'ypred_w{i}'
            if percentile_value is not None:
                result = np.percentile(group[key], percentile_value).astype('float32')
            else:
                result = group[key].max().astype('float32')
            results[key].append(result)

        results['block'].append(name)
        results['x'].append(group.iloc[0]['x'])
        results['y'].append(group.iloc[0]['y'])
        results['cultivar'].append(group.iloc[0]['cultivar'])
        results['ytrue'].append(mean_ytrue)

    
    for name, block in test_df.groupby('block'):
        for _, group in block.groupby(['x', 'y']):
            process_group(name, group)

    #         # Apply the safe_mean function to each row and then calculate the overall mean
    # mean_value = short_test_df['ypred_w15'].apply(safe_mean).mean()
    # closest_percentile = select_closest_percentile(mean_value, mean_percentile_dict)

    # short_test_df['final'] = short_test_df['ypred_w15'].apply(lambda x: apply_percentile(x, closest_percentile))
    # short_test_df
    return pd.DataFrame(results) #short_test_df#,

def calc_test_blocks_range_values(range: str):

    df = pd.read_csv('/data2/hkaman/Imbalance/EXPs/CNNs/EXP_00_lr001_wd05_drop30_vanilla/00_lr001_wd05_drop30_vanilla_test.csv')
    
    blocks_df = df.groupby(by = 'block')

    blocks_name, blocks_cutilvar, blocks_mean_value = [], [], []
    for b, bdf in blocks_df:
        blocks_name.append(b)
        blocks_cutilvar.append(bdf.iloc[0]['cultivar'])
        blocks_mean_value.append(bdf['ytrue'].mean())

    newdf = pd.DataFrame()
    newdf['block'] = blocks_name
    newdf['cultivar'] = blocks_cutilvar
    newdf['mean'] = blocks_mean_value


    low_range_blocks = newdf[newdf['mean'] < 9]
    common_blocks = newdf[(newdf['mean'] >= 9) & (newdf['mean'] < 22)]
    high_range_blocks = newdf[newdf['mean'] >= 22]

    
    if range == 'low': 
        for idx, row in low_range_blocks.iterrows():
            print(f"[C1] LOW EXTEME RANGE: block name = {int(row['block'])} | cultivar type {row['cultivar']} | mean yield value = {row['mean']*HECTARE_TO_ACRE_SCALE:.2f}")
    elif range == 'common':
        for idx, row in common_blocks.iterrows():
            print(f"[C2] COMMON: block name = {int(row['block'])} | cultivar type {row['cultivar']} | mean yield value = {row['mean']*HECTARE_TO_ACRE_SCALE:.2f}")
    elif range == 'high':
        for idx, row in high_range_blocks.iterrows():
            print(f"[C3] High EXTEME RANGE: block name = {int(row['block'])} | cultivar type {row['cultivar']} | mean yield value = {row['mean']*HECTARE_TO_ACRE_SCALE:.2f}")

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

class weight_vis():
    def __init__(self, 
                 method: str, 
                 lds_ks: int = 10, 
                 lds_sigma: int = 8,
                 dw_alpha: float = 3.9,
                 betha: float = 4):

        self.method = method
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        self.dw_alpha = dw_alpha
        self.betha = betha

        self.df = pd.read_csv('/data2/hkaman/Imbalance/EXPs/CNNs/EXP_02_lr001_wd05_LDSinv_10_8/coords/train.csv')

    def plot(self):
        ytrue = self._return_ytrue()
        if self.method == 'lds':
            weights = self._return_lds_weights(ytrue)
        elif self.method == 'cb':
            weights = self._return_cb_weights(ytrue)
        elif self.method == 'dw':
            weights = self._return_dw_weights(ytrue)
        elif self.method == 'ours':
            weights = self._return_extreme_weights(ytrue)

        weight_means, ytrue_counts = self._return_samples_weight_per_bins(ytrue, weights)    
        
        self._plot(ytrue_counts, weight_means)

        return weight_means

    def crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        crop_src = src[:, xcoord:xcoord + 16, ycoord:ycoord + 16, :]
        return crop_src 
    
    def _return_extreme_weights(self, ytrue):
        all_mean_value = np.mean(ytrue)
        # Perform Kernel Density Estimation
        density_lds = self._return_lds_weights(ytrue)

        # density_lds =  RWSampler.cb_prepare_weights(ytrue, 
        #                                 lds_kernel = 'gaussian', 
        #                                 lds_ks = self.lds_ks, 
        #                                 lds_sigma = self.lds_sigma, 
        #                                 betha = 3.9)
        
        density_lds = density_lds/np.max(density_lds)
        weights = ((density_lds) * ((ytrue - all_mean_value)**2)) + 1
        weights = np.array(weights, dtype = np.float32)

        return weights

    def _return_lds_weights(self, ytrue):
        weights = RWSampler.lds_prepare_weights(ytrue, 'inverse', 
                            max_target = 30, 
                            lds = True, 
                            lds_kernel = 'gaussian', 
                            lds_ks = self.lds_ks, 
                            lds_sigma = self.lds_sigma)
        
        weights = np.array(weights, dtype = np.float32)
        return weights
    
    def _return_cb_weights(self, ytrue):

        weights =  RWSampler.cb_prepare_weights(ytrue, 
                                                lds_kernel = 'gaussian', 
                                                lds_ks = self.lds_ks, 
                                                lds_sigma = self.lds_sigma, 
                                                betha = self.betha)
        weights = np.array(weights, dtype = np.float32)
        return weights  
    
    def _return_dw_weights(self, ytrue):

        weights = RWSampler.TargetRelevance(ytrue, alpha = self.dw_alpha).__call__(ytrue)
        assert not np.isnan(weights).any()

        # weights = np.where(weights >= 1, weights, 1)

        return weights

    def _return_ytrue(self):
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

        reshaped_masks = np.reshape(masks, (masks.shape[0]*masks.shape[1]*masks.shape[2]))

        return reshaped_masks
    
    def _plot(self, ytrue_counts, weights):

        fig, axs = plt.subplots(1, 1, figsize=(12, 5))

        sns.set_style("whitegrid", {'axes.grid': False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace=0.01)

        bins_value = np.floor(np.arange(1, 75, 2.5)).astype(int)

        ax1 = sns.barplot(x=bins_value, y=ytrue_counts, color= sns.color_palette()[0], ax=axs) #sns.color_palette()[0]
        axs01 = axs.twinx()
        ax01 = sns.barplot(x=bins_value, y=weights, color='red', alpha=0.6, ax=axs01)

        handles = [mpatches.Patch(facecolor=sns.color_palette()[0], label='Number of Samples'),
                mpatches.Patch(facecolor='red', label='Weight')]
        axs.legend(handles=handles, loc='upper right')

        ax1.set(ylabel='Number of Samples')
        ax01.set(ylabel='Weight')
        ax1.set(xlabel='bin')

        plt.show()
    
    def _return_samples_weight_per_bins(self, masks, weights):
        ytrue_counts, weight_means = [], []
        for i in range(30):
            # Create a mask for the bin
            bin_mask = (masks > i) & (masks <= (i + 1)) if i > 0 else (masks >= i) & (masks <= (i + 1))

            # Apply the mask to count and calculate mean
            ytrue_count = np.count_nonzero(bin_mask)
            weight_mean = np.mean(weights[bin_mask])

            ytrue_counts.append(ytrue_count)
            weight_means.append(weight_mean)

        return weight_means, ytrue_counts
#=========================================================================#
#================================ Classes ================================#
#=========================================================================#

class performance():
    def __init__(self, 
                 exp_name: str):
        self.exp_name = exp_name

        self.exp_output_dir = '/data2/hkaman/Imbalance/EXPs/CNNs/' + 'EXP_' + exp_name 

        self.train_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_train.csv'), index_col=0)
        self.valid_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_valid.csv'), index_col=0)
        self.test_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_test.csv'), index_col=0)

        # self.iter_test_list_df = return_iter_test_dfs(os.path.join(self.exp_output_dir, exp_name))
        # self.modified_quantile_test_df = return_modified_df(self.test_df)
    def scatter_plot(self):

        fig = plt.figure(figsize=(21, 7))
        gs  = gridspec.GridSpec(1, 3)

        train_plot = self._plot_scatter(self.train_df)
        valid_plot = self._plot_scatter(self.valid_df)
        test_plot = self._plot_scatter(self.test_df)

        mg0 = SeabornFig2Grid(train_plot, fig, gs[0])
        mg1 = SeabornFig2Grid(valid_plot, fig, gs[1])
        mg2 = SeabornFig2Grid(test_plot, fig, gs[2])

        gs.tight_layout(fig)
        plt.show()
    
    def time_series_plot(self):
        
        Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
        
        results = pd.DataFrame()
        
        # test_r2_mean, test_mae_mean, test_rmse_mean, test_mape_mean = [], [], [], []
        # test_r2_std, test_mae_std, test_rmse_std, test_mape_std = [], [], [], []
        test_r2_temp, test_mae_temp, test_rmse_temp, test_mape_temp = [], [], [], []
        for w in range(1, 16, 1): 
        
            # for i in range(10):
            test_r2, test_mae, test_rmse, test_mape, _, _ = regression_metrics(self.test_df['ytrue']* HECTARE_TO_ACRE_SCALE, 
                                                                                self.test_df[f'ypred_w{w}']* HECTARE_TO_ACRE_SCALE)
            test_r2_temp.append(test_r2)
            test_mae_temp.append(test_mae)
            test_rmse_temp.append(test_rmse)
            test_mape_temp.append(test_mape)
                
            
            # test_r2_mean.append(np.mean(test_r2_temp))
            # test_mae_mean.append(np.mean(test_mae_temp))
            # test_rmse_mean.append(np.mean(test_rmse_temp))
            # test_mape_mean.append(np.mean(test_mape_temp))
            
            # test_r2_std.append(np.std(test_r2_temp))
            # test_mae_std.append(np.std(test_mae_temp))
            # test_rmse_std.append(np.std(test_rmse_temp))
            # test_mape_std.append(np.std(test_mape_temp))
        
        results['weeks'] = Weeks
        results['R2_mean'] = test_r2_temp
        results['MAE_mean'] = test_mae_temp
        results['RMSE_mean'] = test_rmse_temp
        results['MAPE_mean'] = test_mape_temp
        # results['R2_std'] = test_r2_std
        # results['MAE_std'] = test_mae_std
        # results['RMSE_std'] = test_rmse_std
        # results['MAPE_std'] = test_mape_std

        fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
        metrics = [('R2', 'R2'), ('RMSE', 'RMSE'), ('MAE', 'MAE (t/ha)'), ('MAPE', 'MAPE (%)')]
        axs = axs.flatten() 

        for ax, (metric, label) in zip(axs, metrics):
            ax.plot(results["weeks"], results[f'{metric}_mean'], "-d")
            # ax.fill_between(results["weeks"], results[f'{metric}_mean'] - results[f'{metric}_std'], 
            #                 results[f'{metric}_mean'] + results[f'{metric}_std'], alpha=.2)
            ax.set_ylabel(label)
            ax.set_facecolor('white')
            plt.setp(ax.spines.values(), color='k')

        axs[2].tick_params(axis='x', rotation=45) 
        axs[-1].tick_params(axis='x', rotation=45) 
        axs[-1].legend(loc="upper right")
        plt.show()
        None

        return results
    
    def mape_per_yield_range(self, th1: int, th2: int):

        test_df_ytrue = self.test_df['ytrue'].values * HECTARE_TO_ACRE_SCALE

        len_C1 = len(test_df_ytrue[np.where((test_df_ytrue >= 0) & (test_df_ytrue < th1))])
        len_C2 = len(test_df_ytrue[np.where((test_df_ytrue >= th1) & (test_df_ytrue < th2))])
        len_C3 = len(test_df_ytrue[np.where(test_df_ytrue >= th2)])
        #
        true_labels = self.test_df['ytrue'].values * HECTARE_TO_ACRE_SCALE
        pred_labels = self.test_df['ypred_w15'].values * HECTARE_TO_ACRE_SCALE

        #if i < th1: 
        true_label_C1 = true_labels[np.where((true_labels >= 0) & (true_labels < th1))]
        pred_label_C1 = pred_labels[np.where((true_labels >= 0) & (true_labels < th1))]

        #elif (i >= th1) & (i < th2):
        true_label_C2 = true_labels[np.where((true_labels >= th1) & (true_labels < th2))]
        pred_label_C2 = pred_labels[np.where((true_labels >= th1) & (true_labels < th2))]

        #elif i >= th2: 
        true_label_C3 = true_labels[np.where(true_labels >= th2)]
        pred_label_C3 = pred_labels[np.where(true_labels >= th2)]

        All_R2, All_MAE, All_RMSE, All_MAPE, _, _ = regression_metrics(true_labels, pred_labels)
        print(f"C1 num samples: {len_C1} | C2 num samples: {len_C2} | C3 num samples: {len_C3}")
        C1_R2, C1_MAE, C1_RMSE, C1_MAPE, _, _ = regression_metrics(true_label_C1, pred_label_C1)
        C2_R2, C2_MAE, C2_RMSE, C2_MAPE, _, _ = regression_metrics(true_label_C2, pred_label_C2)
        C3_R2, C3_MAE, C3_RMSE, C3_MAPE, _, _ = regression_metrics(true_label_C3, pred_label_C3)

        print(f"C1 is yield value between 0 and {th1}, C2 is yield value between {th1} and {th2}, and C3 is yield value bigger than {th2}")
        print(f"All: MAE = {All_MAE:.2f}, MAPE = {All_MAPE:.2f} | C1: MAE = {C1_MAE:.2f}, MAPE = {C1_MAPE:.2f} | C2: MAE = {C2_MAE:.2f}, MAPE = {C2_MAPE:.2f} | C3: MAE = {C3_MAE:.2f}, MAPE = {C3_MAPE:.2f}")
        print(f"=========================================================================================================================")
        return [C1_MAE, C1_MAPE, C2_MAE, C2_MAPE, C3_MAE, C3_MAPE]

    def mape_per_bin_plot(self):
        fig, axs = plt.subplots(1, 1, figsize=(16, 4))

        sns.set_style("whitegrid", {'axes.grid': False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace=0.01)
        self.modified_quantile_test_df = self.test_df
        MAPE_Errors, counts = [], []
        for i in range(1, 71, 2):
            if i == 1:
                count = self.modified_quantile_test_df.loc[(self.modified_quantile_test_df['ytrue'] * HECTARE_TO_ACRE_SCALE < (i + 2))]
                Data = self.test_df.loc[(self.test_df['ytrue'] * HECTARE_TO_ACRE_SCALE < (i + 2))]
            elif i >= 68:  
                count = self.modified_quantile_test_df.loc[(self.modified_quantile_test_df['ytrue'] * HECTARE_TO_ACRE_SCALE >= i)]
                Data = self.test_df.loc[(self.test_df['ytrue'] * HECTARE_TO_ACRE_SCALE >= i)]
            else:

                count = self.modified_quantile_test_df.loc[(self.modified_quantile_test_df['ytrue'] * HECTARE_TO_ACRE_SCALE >= i) & (self.modified_quantile_test_df['ytrue'] * HECTARE_TO_ACRE_SCALE < (i + 2))]
                Data = self.test_df.loc[(self.test_df['ytrue'] * HECTARE_TO_ACRE_SCALE >= i) & (self.test_df['ytrue'] * HECTARE_TO_ACRE_SCALE < (i + 2))]
            
            counts.append(len(count))
            if len(Data) == 0 or (i<=10):
                MAPE = 0    
            else:
                MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
            if MAPE > 1:
                MAPE = 0
            MAPE_Errors.append(MAPE * 100)

        # pearson_value = pearsonr(MAPE_Errors, counts)[0]
        bins_value = np.arange(1, 71, 2)  # Adjusted bins value

        ax1 = sns.barplot(x=bins_value, y=counts, color=sns.color_palette()[0], ax=axs)
        axs01 = axs.twinx()
        ax01 = sns.barplot(x=bins_value, y=MAPE_Errors, color=sns.color_palette()[3], alpha=0.5, ax=axs01)

        handles = [mpatches.Patch(facecolor=sns.color_palette()[0], label='Number of Samples'),
                mpatches.Patch(facecolor=sns.color_palette()[3], label='MAPE Error (%)')]
        axs.legend(handles=handles, loc='upper right')

        ax1.set(ylabel='Number of Samples')
        ax01.set(ylabel='MAPE Error (%)')
        ax01.set_ylim(0, 100)
        ax1.set(xlabel='Yield Value (t/ha)')
        # plt.title(r"Pearson Correlation: {:.2f}".format(pearson_value))

        for j, m in enumerate(MAPE_Errors):        
            ax01.text(j ,m, "{:.2f}".format(m), color='k', position = (j - 0.15, m + 1.5), fontsize = 'small', rotation=90)

        plt.show()

    def _plot_scatter(self, df):

        week_pred = 'ypred_w' + str(WEEK_FOR_VIS)

        data = df[['ytrue', week_pred]].rename(columns={week_pred: "ypred"})

        true_values = data['ytrue'] * HECTARE_TO_ACRE_SCALE
        pred_values = data['ypred'] * HECTARE_TO_ACRE_SCALE

        r2, mae, rmse, mape, _, _ = regression_metrics(true_values, pred_values)

        g = sns.jointplot(x = true_values, 
                          y = pred_values, 
                          kind = "hex", 
                          height = 8, 
                          ratio = 4,
                          xlim = [0, MAXIMUM_AXIS_VALUE], ylim=[0, MAXIMUM_AXIS_VALUE], 
                          extent = [0, MAXIMUM_AXIS_VALUE, 0, MAXIMUM_AXIS_VALUE], 
                          gridsize = 100,
                          cmap = PLOT_CMAP, 
                          mincnt = PLOT_MINCNT, 
                          joint_kws = {"facecolor": PLOT_VIS_FACE_COLOR})

        for patch in g.ax_marg_x.patches:
            patch.set_facecolor(PLOT_BOX_FACE_COLOR)

        for patch in g.ax_marg_y.patches:
            patch.set_facecolor(PLOT_BOX_FACE_COLOR)

        g.ax_joint.plot([0, MAXIMUM_AXIS_VALUE], [0, MAXIMUM_AXIS_VALUE], '--r', linewidth=2)

        plt.xlabel('Measured (t/ha)')
        plt.ylabel('Predicted (t/ha)')
        plt.grid(False)

        scores = (r'R^2={:.2f}' + '\n' + r'MAE={:.2f} (t/ha)' + '\n' + r'RMSE={:.2f} (t/ha)' + '\n' + r'MAPE={:.2f} %').format(
            r2, mae, rmse, mape)

        plt.text(1, 74.7, scores, bbox=dict(facecolor = PLOT_VIS_FACE_COLOR, edgecolor = PLOT_BOX_FACE_COLOR, boxstyle = 'round, pad=0.2'),
                fontsize = PLOT_TEXT_FONTSIZE, ha='left', va = 'top')

        return g

class multi_model_timeseries_plot():
    def __init__(self):
        self.weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
        model_files_dict = self._return_full_df_names()
        self.full_df = self._return_full_df(model_files_dict)
        

    def plot(self):
        custom_labels_dict = {
                'EXP_00_lr001_wd05_drop30_vanilla': 'Vanilla',
                'EXP_01_lr001_wd05_resampling': 'CSR',
                'EXP_02_lr001_wd05_drop30_focalr': 'Focal-R',
                'EXP_03_lr001_wd05_LDSinv_10_8': 'LDS',
                'EXP_04_lr001_wd05_DW_3.9': 'Dense Weight',
                'EXP_05_lr001_wd05_CB_3': 'Class Balanced',
                'EXP_06_lr001_wd05_ExW': 'Extreme Weight',
                'EXP_07_lr001_wd05_drop30_yz': 'Yield Zone',
                'EXP_08_lr001_wd05_drop30_resampling_yz': 'Extreme Weight + Yield Zone',
            }
        
        fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
        metrics = [('R2', 'R2'), ('RMSE', 'RMSE'), ('MAE', 'MAE (t/ha)'), ('MAPE', 'MAPE (%)')]
        axs = axs.flatten() 
        # custom_labels = ['Vanilla', 'CSR', 'LDS', 'Class Balanced', 'Dense Weight', 'CSR_Yield Zone']  
        markers = ['o', 'v', '^', '<', '>', 's', '^', 'o', 'v']  
        models = self.full_df['model'].unique()


        for ax, (metric, label) in zip(axs, metrics):
            for model, marker in zip(models, markers):
                custom_label = custom_labels_dict.get(model)
                sns.lineplot(x="weeks", y=f'{metric}_mean', data=self.full_df[self.full_df['model'] == model], 
                            ax=ax, marker=marker, linewidth = 2.5, label=custom_label)
            ax.set_ylabel(label, fontsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_facecolor('white')
            plt.setp(ax.spines.values(), color='k')
            if ax == axs[-1]:  # Show legend only for the last plot
                ax.legend(title='Model', loc="upper right", fontsize=10)
            else:
                ax.legend().remove()

        axs[2].tick_params(axis='x', labelsize=14, rotation=45) 
        axs[3].tick_params(axis='x', labelsize=14, rotation=45)
        plt.show()

    def _return_full_df(self, dict):

        full_df = pd.DataFrame()
        
        list_df = []
        test_r2, test_mae, test_rmse, test_mape = [], [], [], []
        test_r2, test_mae, test_rmse, test_mape = [], [], [], []

        for key, values in dict.items():
            model_df = pd.DataFrame()
            test_r2_mean, test_mae_mean, test_rmse_mean, test_mape_mean = [], [], [], []
            test_r2_std, test_mae_std, test_rmse_std, test_mape_std = [], [], [], []
            test_r2_temp, test_mae_temp, test_rmse_temp, test_mape_temp = [], [], [], []
            df = pd.read_csv(values[0], index_col=0)
            for w in range(1, 16, 1): 
                # for file in values:
                test_r2, test_mae, test_rmse, test_mape, _, _ = regression_metrics(df['ytrue']* HECTARE_TO_ACRE_SCALE, 
                                                                                    df[f'ypred_w{w}']* HECTARE_TO_ACRE_SCALE)
                
                test_r2_temp.append(test_r2)
                test_mae_temp.append(test_mae)
                test_rmse_temp.append(test_rmse)
                test_mape_temp.append(test_mape)
                    
                # test_r2_mean.append(np.mean(test_r2_temp))
                # test_mae_mean.append(np.mean(test_mae_temp))
                # test_rmse_mean.append(np.mean(test_rmse_temp))
                # test_mape_mean.append(np.mean(test_mape_temp))
                
                # test_r2_std.append(np.std(test_r2_temp))
                # test_mae_std.append(np.std(test_mae_temp))
                # test_rmse_std.append(np.std(test_rmse_temp))
                # test_mape_std.append(np.std(test_mape_temp))

            model_df['model'] = 15*[key]
            model_df['weeks'] = self.weeks
            model_df['R2_mean'] = test_r2_temp
            model_df['MAE_mean'] = test_mae_temp
            model_df['RMSE_mean'] = test_rmse_temp
            model_df['MAPE_mean'] = test_mape_temp
            # model_df['R2_std'] = test_r2_std
            # model_df['MAE_std'] = test_mae_std
            # model_df['RMSE_std'] = test_rmse_std
            # model_df['MAPE_std'] = test_mape_std

            list_df.append(model_df)

        full_df = pd.concat(list_df)

        return full_df

    def _return_full_df_names(self):

        base_dir = '/data2/hkaman/Imbalance/EXPs/CNNs'  # Replace with the actual path to your 'CNN' folder
        model_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

        model_files_dict = {}

        for model in model_dirs:
            model_path = os.path.join(base_dir, model)
            csv_files = sorted(glob.glob(os.path.join(model_path, '*test.csv')))
            model_files_dict[model] = csv_files
        
        return model_files_dict

class timeseries_spatial_variability():
    def __init__(
            self, 
            exp_name: str,
    ):

        self.exp_output_dir = '/data2/hkaman/Imbalance/EXPs/CNNs/' + 'EXP_' + exp_name 
        # train_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_train.csv'), index_col=0)
        test_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_test_sp.csv'), index_col=0)
        self.test_df = return_modified_df(test_df, cat = 'extreme')
        # print(self.test_df.shape)
    def multiscale_plot(self, block_name: str, year:int,  min_v: None, max_v: None):

        ytrue10, list_ypred = self.return_rebuild_block_matrix(block_name, year = year)
        
        # 20x20
        ytrue20 = aggregate(ytrue10, 2)
        ypred_20_w15  = aggregate(list_ypred[-1], 2)

        # 40x40
        ytrue40 = aggregate(ytrue10, 4)
        ypred_40_w15  = aggregate(list_ypred[-1], 4)

        # 60x60
        ytrue60 = aggregate(ytrue10, 6)
        ypred_60_w15  = aggregate(list_ypred[-1], 6)

        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        im_true = axs[0, 0].imshow(ytrue10, vmin=min_v, vmax=max_v)
        axs[0, 0].set_title('Yield Observation (10m)', fontsize=16)
        # axs[0, 0].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im_true,  cax=cax)
        im_true.set_clim(min_v, max_v)

        img = axs[0, 1].imshow(list_ypred[-1], vmin=min_v, vmax=max_v)
        _, test_mae, _, test_mape, _, _ = regression_metrics(ytrue10, list_ypred[-1])
        axs[0, 1].set_title(f'Yield Prediction Week 15: \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
        # axs[0, 1].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img,  cax=cax)
        img.set_clim(min_v, max_v)
        
        # distribution plot: 
        ytrue10_vector = ytrue10.flatten()
        ytrue10_vector = ytrue10_vector[ytrue10_vector != -1]

        ypred10_vector = list_ypred[-1].flatten()
        ypred10_vector = ypred10_vector[ypred10_vector != -1]

        ypred20_vector = ypred_20_w15.flatten()
        ypred20_vector = ypred20_vector[ypred20_vector != -1]

        ypred40_vector = ypred_40_w15.flatten()
        ypred40_vector = ypred40_vector[ypred40_vector != -1]

        ypred60_vector = ypred_60_w15.flatten()
        ypred60_vector = ypred60_vector[ypred60_vector != -1]


        data = {
        'Yield (t/ha)': np.concatenate([ytrue10_vector, ypred10_vector, ypred20_vector, ypred40_vector, ypred60_vector]),
        'Matrix': ['Yield Observation']*len(ytrue10_vector) + ['Yield Prediction (10m)']*len(ypred10_vector) + ['Yield Prediction (20m)']*len(ypred20_vector) + ['Yield Prediction (40m)']*len(ypred40_vector) + ['Yield Prediction (60m)']*len(ypred60_vector) 
            }
        df = pd.DataFrame(data)
        
        categories = df['Matrix'].unique()
        for category in categories:
            sns.kdeplot(df[df['Matrix'] == category], x='Yield (t/ha)', fill=True, ax=axs[0, 2], label=category)
        axs[0, 2].legend(loc='upper left')


        img = axs[1, 0].imshow(ypred_20_w15, vmin=min_v, vmax=max_v)
        _, test_mae, _, test_mape, _, _ = regression_metrics(ytrue20, ypred_20_w15)
        axs[1, 0].set_title(f'Yield Prediction Week 15 (20m): \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
        # axs[1, 0].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img,  cax=cax)
        img.set_clim(min_v, max_v)

        img = axs[1, 1].imshow(ypred_40_w15, vmin=min_v, vmax=max_v)
        _, test_mae, _, test_mape, _, _ = regression_metrics(ytrue40, ypred_40_w15)
        axs[1, 1].set_title(f'Yield Prediction Week 15 (40m): \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
        # axs[1, 1].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img,  cax=cax)
        img.set_clim(min_v, max_v)
    
        img = axs[1, 2].imshow(ypred_60_w15, vmin=min_v, vmax=max_v)
        _, test_mae, _, test_mape, _, _ = regression_metrics(ytrue60, ypred_60_w15)
        axs[1, 2].set_title(f'Yield Prediction Week 15 (60m): \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
        # axs[1, 2].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img,  cax=cax)
        img.set_clim(min_v, max_v)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25, wspace=0.1)

    def timeseries_plot(self, block_name:str, year: int,  min_v: None, max_v: None):

        
        list_ytrue, list_ypred = self.return_rebuild_block_matrix(block_name, year = year)


        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(4, 4, figsize=(24, 24))

   
        im_true = axs[0, 0].imshow(list_ytrue, vmin=min_v, vmax=max_v)
        axs[0, 0].set_title('Yield Observation', fontsize=16)
        axs[0, 0].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im_true,  cax=cax)
        im_true.set_clim(min_v, max_v)


        # Plotting predicted images
        for i in range(15):
            row, col = divmod(i + 1, 4)
            img = axs[row, col].imshow(list_ypred[i], vmin=min_v, vmax=max_v)
            _, test_mae, _, test_mape, _, _ = regression_metrics(list_ytrue, list_ypred[i])
            axs[row, col].set_title(f'Yield Prediction Week {i+1}: \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
            axs[row, col].axis('off')  # Turn off axis
            divider = make_axes_locatable(axs[row, col])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(img,  cax=cax)
            img.set_clim(min_v, max_v)

        # Adjust axes visibility for the last row and first column
        rows = [0, 1, 2]
        for i in rows:
            for ax in axs[i, :]:
                ax.axis('on')
                ax.set_xticks([])
                #ax.set_xlabel('X-axis Label')  # Set your x-axis label here
        columns = [1, 2, 3]
        for c in columns:
            for ax in axs[:, c]:
                ax.axis('on')
                ax.set_yticks([])
                #ax.set_ylabel('Y-axis Label')  # Set your y-axis label here
        axs[3, 0].axis('on')
        plt.tight_layout()
        plt.show()

    def plot(self, block_name:str, year: int, min_v: None, max_v: None):

        list_ytrue, list_ypred = self.return_rebuild_block_matrix(block_name, year = year)
        num_years = len(list_ytrue)

        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(1, 4, figsize = (24, 8*num_years))

        # for i in range(num_years):
        img1 = axs[0].imshow(list_ytrue)
        axs[0].set_title('Yield Observation', fontsize = 16)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(img1,  cax=cax)
        img1.set_clim(min_v, max_v)

        img2 = axs[1].imshow(list_ypred[-1])
        axs[1].set_title('Yield Prediction (Week 15)', fontsize = 16)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img2, cax=cax)
        img2.set_clim(min_v, max_v)
        axs[1].get_yaxis().set_visible(False)

        _, test_mae, _, test_mape, _, _ = regression_metrics(list_ytrue[list_ytrue !=- 1], list_ypred[-1][list_ypred[-1] != -1])


        mae_map, mape_map = self.image_mae_mape_map(list_ytrue, list_ypred[-1])

        img3 = axs[2].imshow(mae_map, cmap = 'viridis') #, cmap = 'magma'
        axs[2].set_title(f'MAE Map (t/ha) = {test_mae:.2f}', fontsize = 16)
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img3, cax=cax)
        img3.set_clim(-1, 10)
        axs[2].get_yaxis().set_visible(False)

        img4 = axs[3].imshow(mape_map, cmap = 'viridis') #
        axs[3].set_title(f'MAPE Map (%) = {test_mape:.2f}', fontsize = 16)
        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar3 =fig.colorbar(img4, cax=cax)
        img4.set_clim(-5, 20)
        axs[3].get_yaxis().set_visible(False)

        fig.subplots_adjust(hspace=0.01, wspace=0.01)  # Adjust these values as needed to reduce the space
        fig.tight_layout()

    def variability_plot(self, min_v, max_v):
        list_ytrue, list_ypred = self.return_rebuild_block_matrix()
        num_years = len(list_ytrue)

        w = min([ytrue.shape[0] for ytrue in list_ytrue])
        l = min([ytrue.shape[1] for ytrue in list_ytrue])


        updated_list_ytrue = [mtx[:w, :l] for mtx in list_ytrue]
        stacked_ytrue_matrices = np.stack(updated_list_ytrue)
        ytrue_pixel_wise_variance = np.std(stacked_ytrue_matrices, axis=0)

        updated_list_ypred = [mtx[:w, :l] for mtx in list_ypred]
        stacked_ypred_matrices = np.stack(updated_list_ypred)
        ypred_pixel_wise_variance = np.std(stacked_ypred_matrices, axis=0)

        mae_map_mean, mape_map_mean =[], []
        for i in range(num_years):
            mae_map, mape_map = self.image_mae_mape_map(updated_list_ytrue[i], updated_list_ypred[i])
            mae_map_mean.append(mae_map)
            mape_map_mean.append(mape_map)
        
        mae_pixel_wise_mean = np.mean(mae_map_mean, axis=0)
        mape_pixel_wise_mean = np.mean(mape_map_mean, axis=0)

        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(1, 4, figsize = (24, 8))

        img1 = axs[0].imshow(ytrue_pixel_wise_variance)
        axs[0].set_title('Yield Observation Variability', fontsize = 14)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(img1,  cax=cax)
        img1.set_clim(min_v, max_v)

        img2 = axs[1].imshow(ypred_pixel_wise_variance)
        axs[1].set_title('Yield Prediction Variability', fontsize = 14)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img2, cax=cax)
        img2.set_clim(min_v, max_v)
        axs[1].get_yaxis().set_visible(False)


        img3 = axs[2].imshow(mae_pixel_wise_mean, cmap = 'viridis') 
        axs[2].set_title('MAE Map (t/ha)', fontsize = 14)
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img3, cax=cax)
        img3.set_clim(-1, np.max(mae_pixel_wise_mean))
        axs[2].get_yaxis().set_visible(False)

        img4 = axs[3].imshow(mape_pixel_wise_mean, cmap = 'viridis') #
        axs[3].set_title('MAPE Map (%)', fontsize = 14)
        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar3 =fig.colorbar(img4, cax=cax)
        img4.set_clim(-5, 20)
        axs[3].get_yaxis().set_visible(False)

    def _find_closest_value(self, my_list, target):
        """
        Find the value in my_list that is closest to the target value.
        Args:
        my_list (list): A list of numbers.
        target (float or int): The target value to compare against.

        Returns:
        closest_value: The value from my_list that is closest to the target.
        """
        closest_value = my_list[0]
        minimum_diff = abs(my_list[0] - target)

        for value in my_list:
            diff = abs(value - target)
            if diff < minimum_diff:
                closest_value = value
                minimum_diff = diff

        return closest_value

    def segmentation_plot(self, min_v = None, max_v= None):

        ytrue, ypred = self.return_rebuild_block_matrix()
        ytrue_seg, ypred_seg = self._return_seg_matrix(ytrue, ypred)

        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(1, 4, figsize = (24, 8))

        img1 = axs[0].imshow(ytrue)
        axs[0].set_title('Yield Observation', fontsize = 14)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(img1,  cax=cax)
        img1.set_clim(min_v, max_v)

        img2 = axs[1].imshow(ypred)
        axs[1].set_title('Yield Prediction', fontsize = 14)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img2, cax=cax)
        img2.set_clim(min_v, max_v)
        axs[1].get_yaxis().set_visible(False)

        


        class_labels = ['0', '0-17.297', '17.297-22.239', '22.239-32.123', '32.123-49.42', '>49.42']
        cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple'])
        bounds = [0, 1, 2, 3, 4, 5, 6]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)


        axs[2].imshow(ytrue_seg, cmap=cmap, norm = norm) 
        axs[2].set_title('Yield Zone True', fontsize = 14)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(0.5, len(bounds)), ax = axs[2])
        # cbar.set_ticklabels(class_labels)
        # cbar.set_label('Class')

        axs[3].imshow(ypred_seg, cmap=cmap, norm = norm)
        axs[3].set_title('Yield Zone Pred', fontsize = 14)
        cbar3 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(0.5, len(bounds)), ax = axs[3])
        # cbar3.set_ticklabels(class_labels)
        # cbar3.set_label('Class')

    def _return_seg_matrix(self, ytrue, ypred):
        # Initialize an empty array with the same shape as the image for the segmented output
        segmented_ytrue = np.zeros_like(ytrue)
        segmented_ypred = np.zeros_like(ypred)
        # Class 1: Pixel values <= 7
        A = HECTARE_TO_ACRE_SCALE*7
        B = HECTARE_TO_ACRE_SCALE*9
        C = HECTARE_TO_ACRE_SCALE*13
        D = HECTARE_TO_ACRE_SCALE*20

        class_boundaries = [0, 17.297, 22.239, 32.123, 49.42, 75]

        for i, boundary in enumerate(class_boundaries):
            if i == 0:
                continue  # Skip the first boundary as it is the lower limit for class 1
            segmented_ytrue[ytrue >= class_boundaries[i-1]] = i
            segmented_ypred[ypred >= class_boundaries[i-1]] = i

        # segmented_ytrue[ytrue < A] = 1
        # segmented_ypred[ypred < A] = 1

        # segmented_ytrue[ytrue < A] = 1
        # segmented_ypred[ypred < A] = 1
        # # Class 2: Pixel values > 7 and < 9
        # segmented_ytrue[(ytrue >= A) & (ytrue < B)] = 2
        # segmented_ypred[(ypred >= A) & (ypred < B)] = 2
        # # Class 2: Pixel values > 9 and < 13
        # segmented_ytrue[(ytrue >= B) & (ytrue < C)] = 3
        # segmented_ypred[(ypred >= B) & (ypred < C)] = 3
        #     # Class 2: Pixel values > 13 and < 20
        # segmented_ytrue[(ytrue >= C) & (ytrue < D)] = 4
        # segmented_ypred[(ypred >= C) & (ypred < D)] = 4
        # # Class 3: Pixel values >= 20
        # segmented_ytrue[ytrue >= D] = 5
        # segmented_ypred[ypred >= D] = 5

        return segmented_ytrue, segmented_ypred
    
    def _return_full_list_names(self, block_name):

        years = ['2016', '2017', '2018', '2019']

        timeseries_block_names = [int(str(block_name) + year) for year in years]
        updated_timeseries_block_names = [year for year in timeseries_block_names if year in self.test_df['block'].unique()]
        updated_years = [str(year)[-4:] for year in updated_timeseries_block_names]


        if len(str(block_name)) == 1:
            timeseries_block_fullnames = ['LIV_00' + str(block_name) + '_' + year for year in updated_years]
        elif len(str(block_name)) == 2:
            timeseries_block_fullnames = ['LIV_0' + str(block_name) + '_' + year for year in updated_years]
        elif len(str(block_name)) == 3:
            timeseries_block_fullnames = ['LIV_' + str(block_name) + '_' + year for year in updated_years]

        return updated_timeseries_block_names, timeseries_block_fullnames

    def return_rebuild_block_matrix(self, block_name, year: int):
        if year == 2016:
            year_id = 0
        elif year == 2017:
            year_id = 1
        elif year == 2018:
            year_id = 2
        elif year == 2019:
            year_id = 3


        timeseries_block_names, timeseries_block_fullnames = self._return_full_list_names(block_name)
        timeseries_block_names, timeseries_block_fullnames = timeseries_block_names[year_id], timeseries_block_fullnames[year_id]

        blocks_df = self.test_df.groupby(by = 'block')
        
        # for idx, block in enumerate(timeseries_block_names):

        this_block_df = blocks_df.get_group(timeseries_block_names)
        # print(f"{block}: {this_block_df.shape}")

        res = {key: configs.blocks_size[key] for key in configs.blocks_size.keys() & {timeseries_block_fullnames}}
        list_d = res.get(timeseries_block_fullnames)
        block_x_size = int(list_d[0]/10.0)
        block_y_size = int(list_d[1]/10.0)

        true_out = np.full((block_x_size, block_y_size), -1)  
        pred_out_w1 = np.full((block_x_size, block_y_size), -1)
        pred_out_w2 = np.full((block_x_size, block_y_size), -1)
        pred_out_w3 = np.full((block_x_size, block_y_size), -1)
        pred_out_w4 = np.full((block_x_size, block_y_size), -1)
        pred_out_w5 = np.full((block_x_size, block_y_size), -1) 
        pred_out_w6 = np.full((block_x_size, block_y_size), -1)
        pred_out_w7 = np.full((block_x_size, block_y_size), -1)
        pred_out_w8 = np.full((block_x_size, block_y_size), -1)
        pred_out_w9 = np.full((block_x_size, block_y_size), -1)
        pred_out_w10 = np.full((block_x_size, block_y_size), -1) 
        pred_out_w11 = np.full((block_x_size, block_y_size), -1)
        pred_out_w12 = np.full((block_x_size, block_y_size), -1)
        pred_out_w13 = np.full((block_x_size, block_y_size), -1)
        pred_out_w14 = np.full((block_x_size, block_y_size), -1)
        pred_out_w15 = np.full((block_x_size, block_y_size), -1) 

        for x in range(block_x_size):
            for y in range(block_y_size):
                new = this_block_df.loc[(this_block_df['x'] == x)&(this_block_df['y'] == y)]
                if len(new) > 0:
                    true_out[x, y] = new['ytrue']* HECTARE_TO_ACRE_SCALE
                    pred_out_w1[x, y] = new['ypred_w1']* HECTARE_TO_ACRE_SCALE
                    pred_out_w2[x, y] = new['ypred_w2']* HECTARE_TO_ACRE_SCALE
                    pred_out_w3[x, y] = new['ypred_w3']* HECTARE_TO_ACRE_SCALE
                    pred_out_w4[x, y] = new['ypred_w4']* HECTARE_TO_ACRE_SCALE
                    pred_out_w5[x, y] = new['ypred_w5']* HECTARE_TO_ACRE_SCALE
                    pred_out_w6[x, y] = new['ypred_w6']* HECTARE_TO_ACRE_SCALE
                    pred_out_w7[x, y] = new['ypred_w7']* HECTARE_TO_ACRE_SCALE
                    pred_out_w8[x, y] = new['ypred_w8']* HECTARE_TO_ACRE_SCALE
                    pred_out_w9[x, y] = new['ypred_w9']* HECTARE_TO_ACRE_SCALE
                    pred_out_w10[x, y] = new['ypred_w10']* HECTARE_TO_ACRE_SCALE
                    pred_out_w11[x, y] = new['ypred_w11']* HECTARE_TO_ACRE_SCALE
                    pred_out_w12[x, y] = new['ypred_w12']* HECTARE_TO_ACRE_SCALE
                    pred_out_w13[x, y] = new['ypred_w13']* HECTARE_TO_ACRE_SCALE
                    pred_out_w14[x, y] = new['ypred_w14']* HECTARE_TO_ACRE_SCALE
                    pred_out_w15[x, y] = new['ypred_w15']* HECTARE_TO_ACRE_SCALE

                        # if mode == 'mean':
                        #     pred_out[x, y] = new['ypred_w15'].mean()* HECTARE_TO_ACRE_SCALE
                        # elif mode == 'close': 
                        #     pred_out[x, y] = self._find_closest_value(list(new['ypred_w15']), new['ytrue'].mean())* HECTARE_TO_ACRE_SCALE
                        # elif mode == 'median': 
                        #     pred_out[x, y] = new['ypred_w15'].quantile(0.5)* HECTARE_TO_ACRE_SCALE

        list_ypred = [pred_out_w1, pred_out_w2, pred_out_w3, pred_out_w4, pred_out_w5,
                            pred_out_w6, pred_out_w7, pred_out_w8, pred_out_w9, pred_out_w10,
                            pred_out_w11, pred_out_w12, pred_out_w13, pred_out_w14, pred_out_w15]

        return true_out, list_ypred     

    def image_mae_mape_map(self, ytrue, ypred): 

        w = ytrue.shape[0]
        h = ytrue.shape[1] 

        ytrue_flat = ytrue.ravel()
        ypred_flat = ypred.ravel()
        out_mae    = np.empty_like(ytrue_flat)
        out_mape   = np.empty_like(ytrue_flat)

        for i in range(len(ytrue_flat)):
            if (ytrue_flat[i] == -1) and (ypred_flat[i] == -1):
                out_mae[i]  = -10
                out_mape[i] = -10
            # elif abs(ytrue_flat[i] - ypred_flat[i]) > 11:
            #     out_mae[i]  = -10
            #     out_mape[i] = -10
            else: 
                out_mae[i]  = abs(ytrue_flat[i] - ypred_flat[i])
                out_mape[i] = ((abs(ytrue_flat[i] - ypred_flat[i]))/ytrue_flat[i])*100

        out1 = out_mae.reshape(ytrue.shape[0], ytrue.shape[1])
        out2 = out_mape.reshape(ytrue.shape[0], ytrue.shape[1])

        return out1, out2
