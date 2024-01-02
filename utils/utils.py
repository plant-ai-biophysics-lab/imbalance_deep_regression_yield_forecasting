import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models.configs import blocks_information, blocks_size


#-------------------------------------------------------------------------------------------#
#                                    Vineyard Data Vis                                      #
#-------------------------------------------------------------------------------------------#
def dual_emp_effective_hist_plot(emp, effective):
     
    fig, axs = plt.subplots(1, 2 , figsize = (15, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    bins_value  = np.arange(1, 31, 1)
    ax1 = sns.barplot(x = bins_value, y= emp, color = sns.color_palette()[0], width = 0.9, ax = axs[0])
    ax1.set_title("Emprical Label Distribution", fontsize = 14)
    ax2 = sns.barplot(x = bins_value, y= effective, color = sns.color_palette()[0], width = 0.9, ax = axs[1])
    ax2.set_title("Effective Label Distribution", fontsize = 14)

    ymax_value = max(max(emp), max(effective)) 
    ax1.set_ylim(0, ymax_value)
    ax2.set_ylim(0, ymax_value)
    None

def cs_emp_effective_we_std_hist_plot(emp, effective, w, std):
     
    fig, axs = plt.subplots(2, 2 , figsize = (15, 8))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    bins_value  = np.arange(1, 31, 1)

    ax1 = sns.barplot(x = bins_value, y= emp, color = sns.color_palette()[0], width = 0.9, ax = axs[0, 0])
    ax1.set_title("Emprical Label Distribution", fontsize = 14)
    ax2 = sns.barplot(x = bins_value, y= effective, color = sns.color_palette()[0], width = 0.9, ax = axs[0, 1])
    ax2.set_title("Effective Label Distribution", fontsize = 14)

    ymax_value = max(max(emp), max(effective)) 
    ax1.set_ylim(0, ymax_value)
    ax2.set_ylim(0, ymax_value)

    ax3 = sns.barplot(x = bins_value, y= w, color = sns.color_palette()[0], width = 0.9, ax = axs[1, 0])
    ax3.set_title("Weights", fontsize = 14)
    ax4 = sns.barplot(x = bins_value, y= std, color = sns.color_palette()[0], width = 0.9, ax = axs[1, 1])
    ax4.set_title("STD of bin pixels value", fontsize = 14)


    None

def aggregate2(src, scale):
    
    w = int(src.shape[0]/scale)
    h = int(src.shape[1]/scale)
    mtx = np.full((w, h), -1, dtype=np.float32)
    for i in range(w):
        for j in range(h):   
            mtx[i,j]=np.mean(src[i*scale:(i+1)*scale, j*scale:(j+1)*scale])                    
    return mtx    

def triple_emp_effective_weights_hist_plot(emp, effective, weights, method: str):
     
    fig, axs = plt.subplots(1, 3 , figsize = (21, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    bins_value  = np.arange(1, 70, 1)
    ax1 = sns.barplot(x = bins_value, y = emp, color = sns.color_palette()[0],  ax = axs[0]) #width = 0.9,
    ax1.set_title("Emprical Label Distribution", fontsize = 14)

    ax2 = sns.barplot(x = bins_value, y = effective, color = sns.color_palette()[0], ax = axs[1]) #width = 0.9, 
    ax2.set_title("Effective Label Distribution", fontsize = 14)

    ax3 = sns.barplot(x = bins_value, y = weights, color = sns.color_palette()[0],  ax = axs[2]) #width = 0.9,
    ax3.set_title(r" Weights ({})".format(method), fontsize = 14)

    None

class vineyard_data_visulization():
    def __init__(self, dataset:dict, cultivar: list):
        self.dataset = dataset
        self.cultivar  = cultivar
    
    
    def by_cultivar_label_dist_vis(self): 


        this_cultivar_dict = {k:v for k, v in self.dataset.items() if self.dataset[k]['cultivar'] == self.cultivar[0]}


        fig, axs = plt.subplots(1, 2 , figsize = (12, 4))

        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace = 0.01)


        plot_labels, handles = [], []
        concat_matrix_all_blocks = []


        for k, v in this_cultivar_dict.items():
    
            label_mtx = v['label']
            label_mtx = label_mtx.flatten()
            label_mtx = label_mtx[label_mtx >= 0] 
            concat_matrix_all_blocks.append(label_mtx)


            ax1 = sns.kdeplot(label_mtx, ax = axs[0], palette='Dark2', shade ='fill', legend=True,)
            plot_labels.append(k)

        for i in range(len(plot_labels)):
            handles.append(mpatches.Patch(facecolor = sns.color_palette()[i], label = plot_labels[i]))

        axs[0].legend(handles = handles, loc = 'upper left')
        ax1.set(ylabel='Density')


        concat_matrix_all_blocks = np.concatenate(concat_matrix_all_blocks)

        ax2 = sns.kdeplot(concat_matrix_all_blocks, ax = axs[1], palette='Dark2', shade ='fill')

        fig.suptitle(self.cultivar[0] + ': label distribution')

    def by_cultivar_image_dist_vis(self): 


        this_cultivar_dict = {k:v for k, v in self.dataset.items() if self.dataset[k]['cultivar'] == self.cultivar[0]}

        fig, axs = plt.subplots(1, 2 , figsize = (12, 4))

        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace = 0.01)


        plot_labels, handles = [], []
        for k, v in this_cultivar_dict.items():
            img_mtx = v['image']
            img_mtx = np.nan_to_num(img_mtx)
            img_mtx = img_mtx.flatten()

            ax1 = sns.kdeplot(img_mtx, ax = axs[0], palette='Dark2', shade ='fill', legend=True,)
            plot_labels.append(k)
        for i in range(len(plot_labels)):
            handles.append(mpatches.Patch(facecolor = sns.color_palette()[i], label = plot_labels[i]))

        axs[0].legend(handles = handles, loc = 'upper left')
        ax1.set(ylabel='Density')

        concat_matrix_all_blocks = []
        for k, v in this_cultivar_dict.items():
            concat_matrix_all_blocks.append(v['image'].flatten())

        concat_matrix_all_blocks = np.concatenate(concat_matrix_all_blocks, axis = 0)
        concat_matrix_all_blocks = np.nan_to_num(concat_matrix_all_blocks)

        ax2 = sns.kdeplot(concat_matrix_all_blocks, ax = axs[1], palette='Dark2', shade ='fill')

        fig.suptitle(self.cultivar[0] + ': feature distribution')

    def all_cultivar_label_dist_vis(self): 


        new_dict = {}

        for cul in self.cultivar: 
            this_cultivar_dict = {k:v for k, v in self.dataset.items() if self.dataset[k]['cultivar'] == cul}
            labels = []
            for k2, v2 in this_cultivar_dict.items(): 
                labels.append(v2['label'].flatten())
            labels = np.concatenate(labels)
            labels = labels[labels >= 0]

            new_dict[cul] = {'label': labels} 


        fig, axs = plt.subplots(1, 2 , figsize = (16, 6))

        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace = 0.01)


        plot_labels, handles = [], []
        concat_matrix_all_blocks = []


        for k, v in new_dict.items():
            label_mtx = v['label']
            concat_matrix_all_blocks.append(label_mtx)
            ax1 = sns.kdeplot(label_mtx, ax = axs[0], palette='Dark2', shade ='fill', legend=True,)
            plot_labels.append(k)

        for i in range(len(plot_labels)):
            handles.append(mpatches.Patch(facecolor = sns.color_palette()[i], label = plot_labels[i]))

        axs[0].legend(handles = handles, loc = 'upper left')
        ax1.set(ylabel='Density')

        concat_matrix_all_blocks = np.concatenate(concat_matrix_all_blocks)

        ax2 = sns.kdeplot(concat_matrix_all_blocks, ax = axs[1], palette='Dark2', shade ='fill')

        fig.suptitle('All cultivars label distribution')


    def all_cultivar_image_dist_vis(self): 

        new_dict = {}

        for cul in self.cultivar: 
            this_cultivar_dict = {k:v for k, v in self.dataset.items() if self.dataset[k]['cultivar'] == cul}
            images = []
            for k2, v2 in this_cultivar_dict.items(): 
                images.append(v2['image'].flatten())
            images = np.concatenate(images)
            images = np.nan_to_num(images)

            new_dict[cul] = {'image': images} 

        fig, axs = plt.subplots(1, 2 , figsize = (14, 5))

        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace = 0.01)


        plot_labels, handles = [], []

        for k, v in new_dict.items():
            img_mtx = v['image']
            ax1 = sns.kdeplot(img_mtx, ax = axs[0], palette='Dark2', shade ='fill', legend=True,)
            plot_labels.append(k)

        for i in range(len(plot_labels)):
            handles.append(mpatches.Patch(facecolor = sns.color_palette()[i], label = plot_labels[i]))

        axs[0].legend(handles = handles)
        ax1.set(ylabel='Density')

        concat_matrix_all_blocks = []
        for k, v in new_dict.items():
            concat_matrix_all_blocks.append(v['image'])

        concat_matrix_all_blocks = np.concatenate(concat_matrix_all_blocks)

        ax2 = sns.kdeplot(concat_matrix_all_blocks, ax = axs[1], palette='Dark2', shade ='fill')

        fig.suptitle('All cultivars feature distribution')

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
    
    
    
    res           = {key: blocks_size[key] for key in blocks_size.keys() & {this_block_name}}
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
                pred_out[x, y] = new['ypred_w15'].min()*2.2417
                true_out[x, y] = new['ytrue'].mean()*2.2417


    if aggregation is True: 
        
        print(f"{pred_out.shape}|{true_out.shape}")
        pred_agg = aggregate2(pred_out, scale)
        true_agg = aggregate2(true_out, scale)
        print(f"Agg: {pred_agg.shape}|{true_agg.shape}")
        
        df_agg = pd.DataFrame() 
        
        flat_pred_agg = pred_agg.flatten()
        flat_pred_agg = flat_pred_agg[flat_pred_agg != -1]
        
        flat_true_agg = true_agg.flatten()
        flat_true_agg = flat_true_agg[flat_true_agg != -1] 
              
        df_agg['ytrue'] = flat_true_agg
        df_agg['ypred'] = flat_pred_agg 
        
        return df_agg, true_agg, pred_agg
    else:
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

    #return mape_map
    #plt.savefig('./imgs/B186_1m.png', dpi = 300)


#-------------------------------------------------------------------------------------------#
#                                Data Complexity Measures                                   #
#-------------------------------------------------------------------------------------------# 
class data_complexity_measure():
    def __init__(self, dataframe, proj: str, method: str, basedon: str):

        self.dataset = dataframe
        self.proj    = proj
        self.method  = method 
        self.basedon = basedon

    def return_output(self):
        if self.method == 'F1':
            output = self.MaxFisherDR()

        return output

    def MaxFisherDR(self):
        if self.proj == 'vienyard':
            unique_classes = sorted(self.dataset['cultivar'].unique())

        elif self.proj == 'lettuce':
            unique_classes = sorted(self.dataset['LtID'].unique())

        cls = []
        f1s=[]

        for i in unique_classes:
            for j in unique_classes: 
                if (i != j) and (j > i): 
                    #get the samples of the 2 classes being considered this iteration
                    sample_c1 = df.loc[df['LtID'] == i]
                    sample_c2 = df.loc[df['LtID'] == j]

                    avg_c1 = np.mean(sample_c1['biomass'], 0)
                    avg_c2 = np.mean(sample_c2['biomass'], 0)

                    std_c1 = np.std(sample_c1['biomass'], 0)
                    std_c2 = np.std(sample_c2['biomass'], 0)

                    f1 = ((avg_c1-avg_c2)**2)/(std_c1**2+std_c2**2)

                    f1 = 1/(1+f1)
                    f1s.append(f1)
                    this_combination = r'LtID: {} vs {}'.format(i, j)
                    cls.append(this_combination)

        list_zip = list(zip(cls, f1s))
        f1_val = np.mean(f1s,axis=0)
    
        return f1_val


    def F2(self):
        '''
        Calculates the F2 measure defined in [1]. 
        Uses One vs One method to handle multiclass datasets.
        -----
        Returns:
        f2s (array): The value of f2 measure for each one vs one combination of classes.
        -----
        References:
        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How com-
        plex is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
        '''
        f2s=[]
        #one vs one method
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                maxmax = np.max([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)
                maxmin = np.max([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                minmin = np.min([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                minmax = np.min([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)

                numer=np.maximum(0.0, minmax - maxmin) 
            
                denom=(maxmax - minmin)
        
                f2 = np.prod(numer/denom)
                f2s.append(f2)

        return f2s

