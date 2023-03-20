import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as  mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set(font_scale=1.5)
sns.set_theme(style='white')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
from src import configs




        
def block_true_pred_mtx(df, block_id, aggregation = None, spatial_resolution  = None, scale = None):
    
    name_split = os.path.split(str(block_id))[-1]
    root_name  = name_split.replace(name_split[-4:], '')
    year       = name_split[-4:]
    
    if len(root_name) == 1:
        this_block_name = 'LIV_00' + root_name + '_' + year
    elif len(root_name) == 2:
        this_block_name = 'LIV_0' + root_name + '_' + year
    elif len(root_name) == 3:
        this_block_name = 'LIV_' + root_name + '_' + year
    
    #print(this_block_name)
    blocks_df = df.groupby(by = 'block')
    this_block_df = blocks_df.get_group(block_id)
    
    
    res           = {key: configs.blocks_size[key] for key in configs.blocks_size.keys() & {this_block_name}}
    list_d        = res.get(this_block_name)
    block_x_size  = int(list_d[0]/spatial_resolution)
    block_y_size  = int(list_d[1]/spatial_resolution)
    
    print(this_block_df.shape)
    pred_out = np.full((block_x_size, block_y_size), -1) 
    true_out = np.full((block_x_size, block_y_size), -1)  

    for x in range(block_x_size):
        for y in range(block_y_size):
            
            new            = this_block_df.loc[(this_block_df['x'] == x)&(this_block_df['y'] == y)] 
            if len(new) > 0:
                pred_out[x, y] = new['ypred_w15'].max()#*2.2417
                true_out[x, y] = new['ytrue'].min()#*2.2417

    return this_block_df, true_out, pred_out


def image_mae_mape_map(ytrue, ypred): 

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
        elif abs(ytrue_flat[i] - ypred_flat[i]) > 11:
            out_mae[i]  = -10
            out_mape[i] = -10
        else: 
            out_mae[i]  = abs(ytrue_flat[i] - ypred_flat[i])
            out_mape[i] = ((abs(ytrue_flat[i] - ypred_flat[i]))/ytrue_flat[i])*100

    out1 = out_mae.reshape(ytrue.shape[0], ytrue.shape[1])
    out2 = out_mape.reshape(ytrue.shape[0], ytrue.shape[1])

    return out1, out2


def yield_true_pred_plot(ytrue, ypred, min_v = None, max_v= None):

    #from mpl_toolkits.axes_grid1 import make_axes_locatable
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

    #mae_map = image_subtract(ytrue, ypred) 
    #mape_map = image_subtract(ytrue, ypred) 
    mae_map, mape_map = image_mae_mape_map(ytrue, ypred)
    img3 = axs[2].imshow(mae_map, cmap = 'viridis') #, cmap = 'magma'
    #xlabel_text = (r"($PSNR = {:.2f}$" + ", " + r"$SSIM = {:.2f}$)").format(PSNR, ssim_value)
    axs[2].set_title('MAE Map (t/h)', fontsize = 14)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar2 =fig.colorbar(img3, cax=cax)
    img3.set_clim(-1, np.max(mae_map))
    axs[2].get_yaxis().set_visible(False)
    #axs[2].get_xaxis().set_visible(False)
    #axs[2].set_xlabel(xlabel_text)

    img4 = axs[3].imshow(mape_map, cmap = 'viridis') #
    #xlabel_text = (r"($PSNR = {:.2f}$" + ", " + r"$SSIM = {:.2f}$)").format(PSNR, ssim_value)
    axs[3].set_title('MAPE Map (%)', fontsize = 14)
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar3 =fig.colorbar(img4, cax=cax)
    img4.set_clim(-5, 20)
    axs[3].get_yaxis().set_visible(False)

def agg_pixelovelapp_df_2d(df):
    out = pd.DataFrame()
    
    newdf = df.groupby(["block", "cultivar", "x", "y"]).agg(
        block    = pd.NamedAgg(column="block", aggfunc=np.unique),
        cultivar = pd.NamedAgg(column="cultivar", aggfunc=np.unique),
        x        = pd.NamedAgg(column="x", aggfunc=np.unique),
        y        = pd.NamedAgg(column="y", aggfunc=np.unique),
        ytrue    = pd.NamedAgg(column="ytrue", aggfunc=np.mean),
        ypred_w1    = pd.NamedAgg(column="ypred_w1", aggfunc=np.mean),
        ypred_w2    = pd.NamedAgg(column="ypred_w2", aggfunc=np.mean),
        ypred_w3    = pd.NamedAgg(column="ypred_w3", aggfunc=np.mean),
        ypred_w4    = pd.NamedAgg(column="ypred_w4", aggfunc=np.mean),
        ypred_w5    = pd.NamedAgg(column="ypred_w5", aggfunc=np.mean),
        ypred_w6    = pd.NamedAgg(column="ypred_w6", aggfunc=np.mean),
        ypred_w7    = pd.NamedAgg(column="ypred_w7", aggfunc=np.mean),
        ypred_w8    = pd.NamedAgg(column="ypred_w8", aggfunc=np.mean),
        ypred_w9    = pd.NamedAgg(column="ypred_w9", aggfunc=np.mean),
        ypred_w10    = pd.NamedAgg(column="ypred_w10", aggfunc=np.mean),
        ypred_w11    = pd.NamedAgg(column="ypred_w11", aggfunc=np.mean),
        ypred_w12    = pd.NamedAgg(column="ypred_w12", aggfunc=np.mean),
        ypred_w13    = pd.NamedAgg(column="ypred_w13", aggfunc=np.mean),
        ypred_w14    = pd.NamedAgg(column="ypred_w14", aggfunc=np.mean),
        ypred_w15    = pd.NamedAgg(column="ypred_w15", aggfunc=np.mean),
        
    )
    
    
    out['block'] = newdf['block'].values
    out['cultivar'] = newdf['cultivar'].values
    out['x'] = newdf['x'].values
    out['y'] = newdf['y'].values
    out['ytrue'] = newdf['ytrue'].values
    out['ypred_w1'] = newdf['ypred_w1'].values
    out['ypred_w2'] = newdf['ypred_w2'].values
    out['ypred_w3'] = newdf['ypred_w3'].values
    out['ypred_w4'] = newdf['ypred_w4'].values
    out['ypred_w5'] = newdf['ypred_w5'].values
    out['ypred_w6'] = newdf['ypred_w6'].values
    out['ypred_w7'] = newdf['ypred_w7'].values
    out['ypred_w8'] = newdf['ypred_w8'].values
    out['ypred_w9'] = newdf['ypred_w9'].values
    out['ypred_w10'] = newdf['ypred_w10'].values
    out['ypred_w11'] = newdf['ypred_w11'].values
    out['ypred_w12'] = newdf['ypred_w12'].values
    out['ypred_w13'] = newdf['ypred_w13'].values
    out['ypred_w14'] = newdf['ypred_w14'].values
    out['ypred_w15'] = newdf['ypred_w15'].values


    return out

def xy_vector_generator(x0, y0, wsize):
    x_vector, y_vector = [], []
    
    for i in range(x0, x0+wsize):
        for j in range(y0, y0+wsize):
            x_vector.append(i)
            y_vector.append(j)

    return x_vector, y_vector 

def time_series_eval_csv(pred_npy, blocks_list, wsize = None):
    #blocks_list = get_blocks_from_patches(pred_npy)
    
    OutDF = pd.DataFrame()
    out_ytrue, out_blocks, out_cultivars, out_x, out_y = [], [], [], [], []
    out_ypred_w1, out_ypred_w2,out_ypred_w3,out_ypred_w4,out_ypred_w5 = [], [], [], [], []
    out_ypred_w6, out_ypred_w7,out_ypred_w8,out_ypred_w9,out_ypred_w10 = [], [], [], [], []
    out_ypred_w11, out_ypred_w12,out_ypred_w13,out_ypred_w14,out_ypred_w15 = [], [], [], [], []
    
    for block in blocks_list:  
        
        name_split = os.path.split(block)[-1]
        block_name = name_split.replace(name_split[7:], '')
        root_name  = name_split.replace(name_split[:4], '').replace(name_split[3], '')
        block_id   = root_name
        
        res           = {key: configs.blocks_information[key] for key in configs.blocks_information.keys() & {block_name}}
        list_d        = res.get(block_name)
        cultivar_id   = list_d[1]

        
        for l in range(len(pred_npy)):
            tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
            if len(tb_pred_indices) !=0:   
                for index in tb_pred_indices:

                    x0                = pred_npy[l]['X'][index]
                    y0                = pred_npy[l]['Y'][index]
                    x_vector, y_vector = xy_vector_generator(x0, y0, wsize)
                    out_x.append(x_vector)
                    out_y.append(y_vector)
       
                    tb_ytrue         = pred_npy[l]['ytrue'][index]
                    tb_flatten_ytrue = tb_ytrue.flatten()
                    out_ytrue.append(tb_flatten_ytrue)
                    

                    tb_ypred_w1    = pred_npy[l]['ypred_w1'][index]
                    tb_flatten_ypred_w1 = tb_ypred_w1.flatten()
                    out_ypred_w1.append(tb_flatten_ypred_w1)
                    
                    tb_ypred_w2    = pred_npy[l]['ypred_w2'][index]
                    tb_flatten_ypred_w2 = tb_ypred_w2.flatten()
                    out_ypred_w2.append(tb_flatten_ypred_w2)
                    
                    tb_ypred_w3    = pred_npy[l]['ypred_w3'][index]
                    tb_flatten_ypred_w3 = tb_ypred_w3.flatten()
                    out_ypred_w3.append(tb_flatten_ypred_w3)
                    
                    tb_ypred_w4    = pred_npy[l]['ypred_w4'][index]
                    tb_flatten_ypred_w4 = tb_ypred_w4.flatten()
                    out_ypred_w4.append(tb_flatten_ypred_w4)
                    
                    tb_ypred_w5    = pred_npy[l]['ypred_w5'][index]
                    tb_flatten_ypred_w5 = tb_ypred_w5.flatten()
                    out_ypred_w5.append(tb_flatten_ypred_w5)

                    tb_ypred_w6    = pred_npy[l]['ypred_w6'][index]
                    tb_flatten_ypred_w6 = tb_ypred_w6.flatten()
                    out_ypred_w6.append(tb_flatten_ypred_w6)
                    
                    tb_ypred_w7    = pred_npy[l]['ypred_w7'][index]
                    tb_flatten_ypred_w7 = tb_ypred_w7.flatten()
                    out_ypred_w7.append(tb_flatten_ypred_w7)
                    
                    tb_ypred_w8    = pred_npy[l]['ypred_w8'][index]
                    tb_flatten_ypred_w8 = tb_ypred_w8.flatten()
                    out_ypred_w8.append(tb_flatten_ypred_w8)
                    
                    tb_ypred_w9    = pred_npy[l]['ypred_w9'][index]
                    tb_flatten_ypred_w9 = tb_ypred_w9.flatten()
                    out_ypred_w9.append(tb_flatten_ypred_w9)
                    
                    tb_ypred_w10    = pred_npy[l]['ypred_w10'][index]
                    tb_flatten_ypred_w10 = tb_ypred_w10.flatten()
                    out_ypred_w10.append(tb_flatten_ypred_w10)
                    
                    tb_ypred_w11    = pred_npy[l]['ypred_w11'][index]
                    tb_flatten_ypred_w11 = tb_ypred_w11.flatten()
                    out_ypred_w11.append(tb_flatten_ypred_w11)
                    
                    tb_ypred_w12    = pred_npy[l]['ypred_w12'][index]
                    tb_flatten_ypred_w12 = tb_ypred_w12.flatten()
                    out_ypred_w12.append(tb_flatten_ypred_w12)
                    
                    tb_ypred_w13    = pred_npy[l]['ypred_w13'][index]
                    tb_flatten_ypred_w13 = tb_ypred_w13.flatten()
                    out_ypred_w13.append(tb_flatten_ypred_w13)
                    
                    tb_ypred_w14    = pred_npy[l]['ypred_w14'][index]
                    tb_flatten_ypred_w14 = tb_ypred_w14.flatten()
                    out_ypred_w14.append(tb_flatten_ypred_w14)
                    
                    tb_ypred_w15    = pred_npy[l]['ypred_w15'][index]
                    tb_flatten_ypred_w15 = tb_ypred_w15.flatten()
                    out_ypred_w15.append(tb_flatten_ypred_w15)
                    
                    tb_block_id   = np.array(len(tb_flatten_ytrue)*[block_id], dtype=np.int32)
                    out_blocks.append(tb_block_id)

                    tb_cultivar_id = np.array(len(tb_flatten_ytrue)*[cultivar_id], dtype=np.int8)
                    out_cultivars.append(tb_cultivar_id)


                    
    # agg    
    out_blocks        = np.concatenate(out_blocks)
    out_cultivars     = np.concatenate(out_cultivars)
    out_x             = np.concatenate(out_x)
    out_y             = np.concatenate(out_y)
    out_ytrue         = np.concatenate(out_ytrue)
    out_ypred_w1         = np.concatenate(out_ypred_w1)
    out_ypred_w2         = np.concatenate(out_ypred_w2)
    out_ypred_w3         = np.concatenate(out_ypred_w3)
    out_ypred_w4         = np.concatenate(out_ypred_w4)
    out_ypred_w5         = np.concatenate(out_ypred_w5)
    out_ypred_w6         = np.concatenate(out_ypred_w6)
    out_ypred_w7         = np.concatenate(out_ypred_w7)
    out_ypred_w8         = np.concatenate(out_ypred_w8)
    out_ypred_w9         = np.concatenate(out_ypred_w9)
    out_ypred_w10         = np.concatenate(out_ypred_w10)
    out_ypred_w11         = np.concatenate(out_ypred_w11)
    out_ypred_w12         = np.concatenate(out_ypred_w12)
    out_ypred_w13         = np.concatenate(out_ypred_w13)
    out_ypred_w14         = np.concatenate(out_ypred_w14)
    out_ypred_w15         = np.concatenate(out_ypred_w15)
    
    OutDF['block']    = out_blocks
    OutDF['cultivar'] = out_cultivars
    OutDF['x']        = out_x
    OutDF['y']        = out_y
    OutDF['ytrue']    = out_ytrue
    OutDF['ypred_w1']    = out_ypred_w1
    OutDF['ypred_w2']    = out_ypred_w2
    OutDF['ypred_w3']    = out_ypred_w3
    OutDF['ypred_w4']    = out_ypred_w4
    OutDF['ypred_w5']    = out_ypred_w5
    OutDF['ypred_w6']    = out_ypred_w6
    OutDF['ypred_w7']    = out_ypred_w7
    OutDF['ypred_w8']    = out_ypred_w8
    OutDF['ypred_w9']    = out_ypred_w9
    OutDF['ypred_w10']    = out_ypred_w10
    OutDF['ypred_w11']    = out_ypred_w11
    OutDF['ypred_w12']    = out_ypred_w12
    OutDF['ypred_w13']    = out_ypred_w13
    OutDF['ypred_w14']    = out_ypred_w14
    OutDF['ypred_w15']    = out_ypred_w15
    
    #NewOUtDF = agg_pixelovelapp_df_2d(OutDF)
    
    return OutDF#, NewOUtDF

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
    mean_ytrue = np.mean(ytrue)
    mean_ypred = np.mean(ypred)
    rmse       = np.sqrt(mean_squared_error(ytrue, ypred))
    mape       = mean_absolute_percentage_error(ytrue, ypred)
    r_square   = r2_score(ytrue, ypred)
    mae        = mean_absolute_error(ytrue, ypred)
    
    return [r_square , mae, rmse, mape, mean_ytrue, mean_ypred] 

def eval_on_three_main_label_range_pred(df, th1: int, th2: int):

    true_labels = df['ytrue'].values
    pred_labels = df['ypred_w15'].values


    #for i in range(30):

    #if i < th1: 
    true_label_C1 = true_labels[np.where((true_labels >= 0) & (true_labels <= th1))]
    pred_label_C1 = pred_labels[np.where((true_labels >= 0) & (true_labels <= th1))]

    #elif (i >= th1) & (i < th2):
    true_label_C2 = true_labels[np.where((true_labels > th1) & (true_labels < th2))]
    pred_label_C2 = pred_labels[np.where((true_labels > th1) & (true_labels < th2))]

    #elif i >= th2: 
    true_label_C3 = true_labels[np.where(true_labels >= th2)]
    pred_label_C3 = pred_labels[np.where(true_labels >= th2)]

    All_R2, All_MAE, All_RMSE, All_MAPE, _, _ = regression_metrics(true_labels, pred_labels)
    print(f"C1 num samples: {len(true_label_C1)} | C2 num samples: {len(true_label_C2)} | C3 num samples: {len(true_label_C3)} ")
    C1_R2, C1_MAE, C1_RMSE, C1_MAPE, _, _ = regression_metrics(true_label_C1, pred_label_C1)
    C2_R2, C2_MAE, C2_RMSE, C2_MAPE, _, _ = regression_metrics(true_label_C2, pred_label_C2)
    C3_R2, C3_MAE, C3_RMSE, C3_MAPE, _, _ = regression_metrics(true_label_C3, pred_label_C3)

    print(f"C1 is yield value between 0 and {th1}, C2 is yield value between {th1} and {th2}, and C3 is yield value bigger than {th2}")
    print(f"All: MAE = {All_MAE:.2f}, MAPE = {All_MAPE:.2f} | C1: MAE = {C1_MAE:.2f}, MAPE = {C1_MAPE*100:.2f} | C2: MAE = {C2_MAE:.2f}, MAPE = {C2_MAPE*100:.2f} | C3: MAE = {C3_MAE:.2f}, MAPE = {C3_MAPE*100:.2f}")
    print(f"=========================================================================================================================")
    return [C1_MAE, C1_MAPE, C2_MAE, C2_MAPE, C3_MAE, C3_MAPE]

def eval_on_extreme_main_label_range_pred(df, th1: int, th2: int):

    true_labels = df['ytrue'].values
    pred_labels = df['ypred_w15'].values


    #for i in range(30):

    #if i < th1: 
    true_label_C1 = true_labels[np.where((true_labels >= 0) & (true_labels <= th1))]
    pred_label_C1 = pred_labels[np.where((true_labels >= 0) & (true_labels <= th1))]

    #elif i >= th2: 
    true_label_C3 = true_labels[np.where(true_labels >= th2)]
    pred_label_C3 = pred_labels[np.where(true_labels >= th2)]


    #elif (i >= th1) & (i < th2):
    true_label_Cm = true_labels[np.where((true_labels > th1) & (true_labels < th2))]
    pred_label_Cm = pred_labels[np.where((true_labels > th1) & (true_labels < th2))]

    true_label_ex = []
    true_label_ex.append(true_label_C1)
    true_label_ex.append(true_label_C3)
    true_label_ex = np.concatenate(true_label_ex)
    pred_label_ex = []
    pred_label_ex.append(pred_label_C1)
    pred_label_ex.append(pred_label_C3)
    pred_label_ex = np.concatenate(pred_label_ex)


    All_R2, All_MAE, All_RMSE, All_MAPE, _, _ = regression_metrics(true_labels, pred_labels)
    print(f"Majority range yield: {len(true_label_Cm)} | Extreme yield value: {len(true_label_ex)}")

    Cm_R2, Cm_MAE, Cm_RMSE, Cm_MAPE, _, _ = regression_metrics(true_label_Cm, pred_label_Cm)
    Cex_R2, Cex_MAE, Cex_RMSE, Cex_MAPE, _, _ = regression_metrics(true_label_ex, pred_label_ex)

    #print(f"C1 is yield value between 0 and {th1}, C2 is yield value between {th1} and {th2}, and C3 is yield value bigger than {th2}")
    print(f"All: MAE = {All_MAE:.2f}, MAPE = {All_MAPE:.2f} | Cm: MAE = {Cm_MAE:.2f}, MAPE = {Cm_MAPE*100:.2f} | Cex: MAE = {Cex_MAE:.2f}, MAPE = {Cex_MAPE*100:.2f}")
    print(f"==============================================================================================================")
    return [Cm_MAE, Cm_MAPE, Cex_MAE, Cex_MAPE]

def Erroe_hist_visulization(df):

    fig, axs = plt.subplots(1, 2 , figsize = (15, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    ax1 = sns.histplot(df, x = 'ytrue', bins = 30, ax = axs[0])
    ax1.set_title("Yield observation distribution")
    MAPE_Errors, counts = [], []
    for i in range(30):
        Data  = df.loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))] 
        counts.append(len(Data))
        MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
        if MAPE > 1: 
            MAPE = 1
        MAPE_Errors.append(MAPE*100)

    pearson_value = pearsonr(MAPE_Errors, counts)[0]
    bins_value  = np.arange(1, 31, 1)
    ax2 = sns.barplot(x = bins_value, y= MAPE_Errors, color = sns.color_palette()[0], width = 0.9, ax = axs[1])



    ax2.set(ylabel='MAPE Error')
    ax2.set(xlabel='yture')
    ax2.set_title(r"Pearson Correlation:{:.2f}".format(pearson_value))

    None

def Erroe_hist_visulization_V2(df):

    fig, axs = plt.subplots(1, 1 , figsize = (12, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    MAPE_Errors, counts = [], []
    for i in range(1, 30):
        if i == 1:
            Data  = df.loc[(df['ytrue'] < (i+1))]
        elif i == 29: 
            Data  = df.loc[(df['ytrue'] >= (i))]
        else: 
            Data  = df.loc[(df['ytrue'] >= i) & (df['ytrue'] < (i+1))] 
        counts.append(len(Data))
        MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
        if MAPE > 1: 
            MAPE = 1
        MAPE_Errors.append(MAPE*100)

    pearson_value = pearsonr(MAPE_Errors, counts)[0]
    bins_value  = np.arange(1, 30, 1)

    ax1  = sns.barplot(x = bins_value, y= counts, color = sns.color_palette()[0], width = 0.9, ax = axs)
    axs01 = axs.twinx()
    ax01 = sns.barplot(x = bins_value, y= MAPE_Errors, color = sns.color_palette()[3], alpha = 0.5, width = 0.9, ax = axs01)


    handles = [mpatches.Patch(facecolor = sns.color_palette()[0], label = 'Number of Samples'),
    mpatches.Patch(facecolor = sns.color_palette()[3], label = 'MAPE Error')
    ]
    axs.legend(handles = handles, loc = 'upper right')


    ax1.set(ylabel='Number of Samples')
    ax01.set(ylabel='MAPE Error')
    ax1.set(xlabel='yield value')
    plt.title(r"Pearson Correlation:{:.2f}".format(pearson_value))

    None

def time_series_evaluation_plots(train, val, test, fig_save_name): 
    
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
    results = pd.DataFrame()
    train_r2_list, train_mae_list, train_rmse_list, train_mape_list = [], [], [], [] 
    val_r2_list, val_mae_list, val_rmse_list, val_mape_list         = [], [], [], []
    test_r2_list, test_mae_list, test_rmse_list, test_mape_list     = [], [], [], []
    
    for i in range(len(Weeks)): 
        train_r2, train_mae, train_rmse, train_mape, _, _ = regression_metrics(train['ytrue'], train.iloc[:, i+1])
        train_r2_list.append(train_r2)
        train_mae_list.append(train_mae)
        train_rmse_list.append(train_rmse)
        train_mape_list.append(train_mape)
        
        val_r2, val_mae, val_rmse, val_mape, _, _         = regression_metrics(val['ytrue'], val.iloc[:, i+1])
        val_r2_list.append(val_r2)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)
        val_mape_list.append(val_mape)
        
        test_r2, test_mae, test_rmse, test_mape, _, _     = regression_metrics(test['ytrue'], test.iloc[:, i+1])
        test_r2_list.append(test_r2)
        test_mae_list.append(test_mae)
        test_rmse_list.append(test_rmse)
        test_mape_list.append(test_mape)
    
    results['weeks']      = Weeks
    results['Train_R2']   = train_r2_list
    results['Valid_R2']   = val_r2_list
    results['Test_R2']    = test_r2_list
    results['Train_MAE']  = train_mae_list
    results['Valid_MAE']  = val_mae_list
    results['Test_MAE']   = test_mae_list
    results['Train_RMSE'] = train_rmse_list
    results['Valid_RMSE'] = val_rmse_list
    results['Test_RMSE']  = test_rmse_list
    results['Train_MAPE'] = train_mape_list
    results['Valid_MAPE'] = val_mape_list
    results['Test_MAPE']  = test_mape_list
 

    #plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(2, 2, figsize = (20, 10), sharex=True)


    axs[0, 0].plot(results["weeks"], results["Train_R2"], "-o")
    axs[0, 0].plot(results["weeks"], results["Valid_R2"], "-*")
    axs[0, 0].plot(results["weeks"], results["Test_R2"],  "-d")
    axs[0, 0].set_ylabel('R2')
    axs[0, 0].set_facecolor('white')
    plt.setp(axs[0, 0].spines.values(), color='k')
    
    axs[0, 1].plot(results["weeks"], results["Train_RMSE"], "-o")
    axs[0, 1].plot(results["weeks"], results["Valid_RMSE"], "-*")
    axs[0, 1].plot(results["weeks"], results["Test_RMSE"],  "-d")
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].set_facecolor('white')
    plt.setp(axs[0, 1].spines.values(), color='k')
    
    axs[1, 0].plot(results["weeks"], results["Train_MAE"], "-o")
    axs[1, 0].plot(results["weeks"], results["Valid_MAE"], "-*")
    axs[1, 0].plot(results["weeks"], results["Test_MAE"],  "-d")
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].set_ylabel('MAE (ton/ac)')
    axs[1, 0].set_facecolor('white')
    plt.setp(axs[1, 0].spines.values(), color='k')
    
    axs[1, 1].plot(results["weeks"], results["Train_MAPE"], "-o", label = 'train')
    axs[1, 1].plot(results["weeks"], results["Valid_MAPE"], "-*", label = 'valid')
    axs[1, 1].plot(results["weeks"], results["Test_MAPE"],  "-d", label = 'test')
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].set_facecolor('white')
    plt.setp(axs[1, 1].spines.values(), color='k')
    axs[1, 1].legend(loc="upper right")
    
    
    

    plt.savefig(fig_save_name, dpi = 300)
    
    return results 

def train_val_test_satterplot(train_df, valid_df, test_df, week = None, cmap  = None, mincnt = None, fig_save_name = None):

    
    if week == None: 
        week_pred = 'ypred'
    else: 
        week_pred = 'ypred_w' + str(week)

    w_train_e1  = train_df[['ytrue', week_pred]]
    w_train_e1  = w_train_e1.rename(columns={week_pred: "ypred"})
    tarin_true  = w_train_e1['ytrue']
    train_pred  = w_train_e1['ypred']
    tr_r2, tr_mae, tr_rmse, tr_mape, _,_ = regression_metrics(tarin_true, train_pred)

    TR = sns.jointplot(x=tarin_true, y=train_pred, kind="hex", height=8, ratio=4,  
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap , mincnt=mincnt, joint_kws={"facecolor": 'white'})#,,  marginal_kws = dict(bins = np.arange(0, 50000))

    for patch in TR.ax_marg_x.patches:
        patch.set_facecolor('grey')

    for patch in TR.ax_marg_y.patches:
        patch.set_facecolor('grey')

    TR.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac) - Train')
    plt.ylabel('Predict (ton/ac)')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(tr_r2, tr_mae, tr_rmse, tr_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Train Data')
    #========================================================
    w_valid_e1  = valid_df[['ytrue', week_pred]]
    w_valid_e1  = w_valid_e1.rename(columns={week_pred: "ypred"})
    valid_true = w_valid_e1['ytrue']
    valid_pred = w_valid_e1['ypred']
    val_r2, val_mae, val_rmse, val_mape, _,_ = regression_metrics(valid_true, valid_pred)

    Va = sns.jointplot(x = valid_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = 'viridis', mincnt = mincnt) #palette ='flare', ,   
    for patch in Va.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Va.ax_marg_y.patches:
        patch.set_facecolor('grey')


    Va.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)
    plt.xlabel('Measured (ton/ac) - Validation')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(val_r2, val_mae, val_rmse, val_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Validation Data')
    #========================================================


    w_test_e1  = test_df[['ytrue', week_pred]]
    w_test_e1  = w_test_e1.rename(columns={week_pred: "ypred"})
    test_true = w_test_e1['ytrue']
    valid_pred = w_test_e1['ypred']

    test_r2, test_mae, test_rmse, test_mape, _,_ = regression_metrics(test_true, valid_pred)

    Te = sns.jointplot(x=test_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap, mincnt = mincnt) #palette ='flare', ,  
    for patch in Te.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Te.ax_marg_y.patches:
        patch.set_facecolor('grey') 

    Te.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac)')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(test_r2, test_mae, test_rmse, test_mape)
    plt.legend([extra], [scores], loc='upper left')



    fig = plt.figure(figsize=(21, 7))
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(TR, fig, gs[0])
    mg1 = SeabornFig2Grid(Va, fig, gs[1])
    mg2 = SeabornFig2Grid(Te, fig, gs[2])


    gs.tight_layout(fig)
    #gs.update(top=0.7)
    #plt.savefig(fig_save_name, dpi = 300)
    plt.show()

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

def ScenarioEvaluation2D(pred_npy):

    OutDF = pd.DataFrame()
    out_ytrue= []
    out_ypred_w1, out_ypred_w2, out_ypred_w3, out_ypred_w4, out_ypred_w5, out_ypred_w6, out_ypred_w7, out_ypred_w8 = [],[],[],[],[],[],[],[]
    out_ypred_w9, out_ypred_w10, out_ypred_w11, out_ypred_w12, out_ypred_w13, out_ypred_w14, out_ypred_w15 = [],[],[],[],[],[],[]
  
    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
            
        ytrue     = pred_npy[l]['ytrue']
        ytrue_flat = ytrue.flatten()
        out_ytrue.append(ytrue_flat)
        
        ypred_w1  = pred_npy[l]['ypred_w1']
        ypred_w1_flat = ypred_w1.flatten()
        out_ypred_w1.append(ypred_w1_flat)
        
        ypred_w2  = pred_npy[l]['ypred_w2']
        ypred_w2_flat = ypred_w2.flatten()
        out_ypred_w2.append(ypred_w2_flat)
        
        ypred_w3  = pred_npy[l]['ypred_w3']
        ypred_w3_flat = ypred_w3.flatten()
        out_ypred_w3.append(ypred_w3_flat)
        
        ypred_w4  = pred_npy[l]['ypred_w4']
        ypred_w4_flat = ypred_w4.flatten()
        out_ypred_w4.append(ypred_w4_flat)
        
        ypred_w5  = pred_npy[l]['ypred_w5']
        ypred_w5_flat = ypred_w5.flatten()
        out_ypred_w5.append(ypred_w5_flat)
        
        ypred_w6  = pred_npy[l]['ypred_w6']
        ypred_w6_flat = ypred_w6.flatten()
        out_ypred_w6.append(ypred_w6_flat)
        
        ypred_w7  = pred_npy[l]['ypred_w7']
        ypred_w7_flat = ypred_w7.flatten()
        out_ypred_w7.append(ypred_w7_flat)
        
        ypred_w8  = pred_npy[l]['ypred_w8']
        ypred_w8_flat = ypred_w8.flatten()
        out_ypred_w8.append(ypred_w8_flat)
        
        ypred_w9  = pred_npy[l]['ypred_w9']
        ypred_w9_flat = ypred_w9.flatten()
        out_ypred_w9.append(ypred_w9_flat)
        
        ypred_w10  = pred_npy[l]['ypred_w10']
        ypred_w10_flat = ypred_w10.flatten()
        out_ypred_w10.append(ypred_w10_flat)
        
        ypred_w11  = pred_npy[l]['ypred_w11']
        ypred_w11_flat = ypred_w11.flatten()
        out_ypred_w11.append(ypred_w11_flat)
        
        ypred_w12  = pred_npy[l]['ypred_w12']
        ypred_w12_flat = ypred_w12.flatten()
        out_ypred_w12.append(ypred_w12_flat)
        
        ypred_w13  = pred_npy[l]['ypred_w13']
        ypred_w13_flat = ypred_w13.flatten()
        out_ypred_w13.append(ypred_w13_flat)
        
        ypred_w14  = pred_npy[l]['ypred_w14']
        ypred_w14_flat = ypred_w14.flatten()
        out_ypred_w14.append(ypred_w14_flat)
        
        ypred_w15  = pred_npy[l]['ypred_w15']
        ypred_w15_flat = ypred_w15.flatten()
        out_ypred_w15.append(ypred_w15_flat)
      


            
    out_ytrue = np.concatenate(out_ytrue)#.astype(np.float16)
    out_ypred_w1 = np.concatenate(out_ypred_w1) 
    out_ypred_w2 = np.concatenate(out_ypred_w2) 
    out_ypred_w3 = np.concatenate(out_ypred_w3) 
    out_ypred_w4 = np.concatenate(out_ypred_w4) 
    out_ypred_w5 = np.concatenate(out_ypred_w5) 
    out_ypred_w6 = np.concatenate(out_ypred_w6) 
    out_ypred_w7 = np.concatenate(out_ypred_w7) 
    out_ypred_w8 = np.concatenate(out_ypred_w8) 
    out_ypred_w9 = np.concatenate(out_ypred_w9) 
    out_ypred_w10 = np.concatenate(out_ypred_w10) 
    out_ypred_w11 = np.concatenate(out_ypred_w11) 
    out_ypred_w12 = np.concatenate(out_ypred_w12) 
    out_ypred_w13 = np.concatenate(out_ypred_w13) 
    out_ypred_w14 = np.concatenate(out_ypred_w14) 
    out_ypred_w15 = np.concatenate(out_ypred_w15) 

    
    OutDF['ytrue'] = out_ytrue
    OutDF['ypred_w1'] = out_ypred_w1
    OutDF['ypred_w2'] = out_ypred_w2 
    OutDF['ypred_w3'] = out_ypred_w3 
    OutDF['ypred_w4'] = out_ypred_w4 
    OutDF['ypred_w5'] = out_ypred_w5 
    OutDF['ypred_w6'] = out_ypred_w6 
    OutDF['ypred_w7'] = out_ypred_w7 
    OutDF['ypred_w8'] = out_ypred_w8 
    OutDF['ypred_w9'] = out_ypred_w9 
    OutDF['ypred_w10'] = out_ypred_w10 
    OutDF['ypred_w11'] = out_ypred_w11 
    OutDF['ypred_w12'] = out_ypred_w12 
    OutDF['ypred_w13'] = out_ypred_w13 
    OutDF['ypred_w14'] = out_ypred_w14 
    OutDF['ypred_w15'] = out_ypred_w15


    return OutDF

def return_samples_error_per_bins(df): 

    MAPE_Errors, counts = [], []
    for i in range(30):
        if i == 0:
            Data  = df.loc[(df['ytrue'] >= i) & (df['ytrue'] <= (i+1))]
        else: 
            Data  = df.loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))] 
        counts.append(len(Data))
        if len(Data) == 0: 
            MAPE = 0
        else:
            MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
        if MAPE > 1: 
            MAPE = 1
        MAPE_Errors.append(MAPE*100)

    return counts, MAPE_Errors

def return_samples_weight_per_bins(df): 
    ytrue_counts, weight_means = [], []
    for i in range(30):
            if i == 0:
                    yture_count = df.loc[(df['ytrue'] >= i) & (df['ytrue'] <= (i+1))]
                    weight_mean = df['weights'].loc[(df['ytrue'] >= i) & (df['ytrue'] <= (i+1))].mean()
            
            else: 
                    yture_count = df.loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))] 
                    weight_mean = df['weights'].loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))].mean()

            ytrue_counts.append(len(yture_count))
            weight_means.append(weight_mean)

    return weight_means, ytrue_counts