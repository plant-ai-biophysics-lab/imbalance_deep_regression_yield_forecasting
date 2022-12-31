import os, os.path
import copy
import math 
from operator import itemgetter, ne
from typing import overload
from numpy.core.numerictypes import ScalarType
import pandas as pd
import numpy as np
from numpy.core.fromnumeric import shape, transpose, var
from scipy.sparse import data
from scipy.spatial.kdtree import distance_matrix
from sklearn.cluster import KMeans
import sklearn
import scipy.spatial
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cv2
import rasterio 
from rasterio.plot import show, show_hist
from rasterio.mask import mask
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
from rasterio.merge import merge
import matplotlib.patches as  mpatches
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr

np.random.seed(0)

#-------------------------------------------------------------------------------------------#
#                                      Data Generation                                     #
#-------------------------------------------------------------------------------------------#

class hyperspectral_label_mtx_gen():

    def __init__(self, img_dir, label_dir, 
                            spatial_resolution: int):  
        
        self.img_dir   = img_dir
        self.lable_dir = label_dir
        self.spatial_resolution = spatial_resolution

        #self.hyper_img, _ = hyper_map_mosaic(self.img_dir, save = True)

    def block_image_label_gen(self):


        block_full_dict = {}
        inner_block_dict = {}


        label_names = sorted(os.listdir(self.lable_dir))
        img_names   = os.listdir(self.img_dir)

        format = ".tif"

        for img in img_names:
            if img.endswith(format):
                for name in label_names: 
                    if name.endswith(format):
                        name_split = os.path.split(name)[-1]
                        name_root = name_split.replace(name_split[-4:], '')

                        label_file_name = self.lable_dir + '/' + name  

                        label_src = rasterio.open(label_file_name)
                        label_data = label_src.read()
                        label_data = np.moveaxis(label_data, 0, 2)
                        # convert all the non values including the background to -1 
                        label_data[label_data<0] = -1 
                        # adding one more dimension to be able for concatenation 
                        h = label_src.height # hight of label image
                        w = label_src.width  # width of label image 
                        slice_ = (slice(0, h), slice(0, w))
                        window_slice = windows.Window.from_slices(*slice_)
                        #print(window_slice)
                        # Window to list of index (row,col) 
                        pol_index = self.to_index(window_slice)
                        # Convert list of index (row,col) to list of coordinates of lat and long 
                        pol = [list(label_src.transform*p) for p in self.reverse_coordinates(pol_index)]
                        roi_polygon_src_coords = warp.transform_geom(label_src.crs,
                                                        {'init': 'epsg:32611'},                                          
                                                        {"type": "Polygon",
                                                        "coordinates": [pol]})

                        # update the label_data based on desired spatial resolution, also addign dimension to the image for concatenation: 
                        
                        new_label_data, new_h, new_w = self.label_data_interpolation(label_data, self.spatial_resolution)
                        # generating hyperspectral images after cropping, normalizing between (0, 255):  
                        img_full_name          = self.img_dir + '/' + img  
                        hyper_cropped_img, _   = self.hyper_img_crop(img_full_name, roi_polygon_src_coords)
                        #print(f"image size: {hyper_cropped_img.shape} | label size: {new_label_data.shape}")
                        hyper_cropped_img   = np.nan_to_num(hyper_cropped_img)
                        # check for consistency with label image size
                        hyper_cropped_img   = hyper_cropped_img[:, 0:new_h, 0:new_w]
                        # normalize the block 
                        hyper_cropped_img_norm = self.normalization_src(hyper_cropped_img)
                        
                        res = {key: blocks_information[key] for key in blocks_information.keys() & {name_root[:-5]}}
                        list_d = res.get(name_root[:-5])

                        block_variety = list_d[0]
                        block_rw      = list_d[2]
                        block_sp      = list_d[3]
                        block_trellis = list_d[5]

                        inner_block_dict = {'image': hyper_cropped_img_norm,
                                            'label': new_label_data,
                                            'cultivar': block_variety, 
                                            'trellis': block_trellis,
                                            'RS': block_sp,
                                            'WS': block_rw}

                        block_full_dict[name_root] = inner_block_dict


        return block_full_dict

    def normalization_src(self, src): 

        scaled_img = None

        for band in range(src.shape[0]):
            this_scaled_band = 255*(src[band, :,:] / np.max(src[band, :,:]))#.astype(np.uint8)
            this_scaled_band = np.expand_dims(this_scaled_band, axis = 0)

            if scaled_img is None: 
                scaled_img = this_scaled_band
            else: 
                scaled_img = np.concatenate((scaled_img, this_scaled_band), axis = 0) 
        
        return scaled_img


    def to_index(self, wind_):
        """
        Generates a list of index (row,col): [[row1,col1],[row2,col2],[row3,col3],[row4,col4],[row1,col1]]
        """
        return [[wind_.row_off,wind_.col_off],
                [wind_.row_off,wind_.col_off+wind_.width],
                [wind_.row_off+wind_.height,wind_.col_off+wind_.width],
                [wind_.row_off+wind_.height,wind_.col_off],
                [wind_.row_off,wind_.col_off]]

    def reverse_coordinates(self, pol):
        """
        Reverse the coordinates in pol
        Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
        Returns [[y1,x1],[y2,x2],...,[yN,xN]]
        """
        #return [list(f[-1::-1]) for f in pol]
        list = []
        for f in pol:
            f2 = f[-1::-1]
            list.append(f2)
        return list

    def label_data_interpolation(self, label_data, spatial_resolution): 

        if spatial_resolution == 1:
            label_data = np.expand_dims(label_data, axis = 0)
            #label_data = np.expand_dims(label_data, axis = -1)

        elif spatial_resolution !=1: 
            scale = 1/spatial_resolution
            label_data = cv2.resize(label_data, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
            label_data = np.expand_dims(label_data, axis = 0)

        h = label_data.shape[1]
        w = label_data.shape[2]

        return label_data, h, w

    def hyper_img_crop(self, img, polygon):

        with rasterio.open(img) as inimg:
            in_cropped, out_transform_in = mask(inimg,
            [polygon],crop=True)
            in_cropped_meta = inimg.meta.copy()
            in_cropped_meta.update({"driver": "GTiff",
                "height": in_cropped.shape[1],
                "width": in_cropped.shape[2], 
                "transform": out_transform_in})

        return in_cropped, in_cropped_meta


class dataframe_split_csv_gen():

    def __init__(self, dataset_dict, img_size: int, offset: int, cultivar_list: list):

        if cultivar_list is None: 
            self.cultivar_list = ['MALVASIA_BIANCA', 'MUSCAT_OF_ALEXANDRIA', 'CABERNET_SAUVIGNON',
                                'SYMPHONY', 'PINOT_GRIS', 'PINOT_NOIR',
                                'MERLOT', 'CHARDONNAY', 'SYRAH', 'RIESLING']

        self.dataset_dict = dataset_dict 
        self.img_size = img_size
        self.offset = offset
        

    def hyper_df_csv(self): 

        df = pd.DataFrame()

        Block, Cultivar, CID, Trellis, TID, RW, SP, P_means = [], [], [], [], [], [], [], []
        YEAR, X_COOR, Y_COOR = [], [], []
        
        
        
        generated_cases = 0
        removed_cases = 0 
        
        
        for block, values in self.dataset_dict.items():

            root_name     = block[0:7]
            year          = block[8:]
            
            res           = {key: blocks_information[key] for key in blocks_information.keys() & {root_name}}
            list_d        = res.get(root_name)
            block_variety = list_d[0]
            block_id      = list_d[1]
            block_rw      = list_d[2]
            block_sp      = list_d[3]
            block_trellis = list_d[5]
            block_tid     = list_d[6]

            label_npy = values['label']
            label     = label_npy[0,:,:]
            width, height = label.shape[1], label.shape[0]
            
            
            for i in range(0, height-self.img_size, self.offset):
                for j in range(0, width-self.img_size, self.offset):
                    crop_label = label[i:i+ self.img_size, j:j+self.img_size]
                    
                    if np.any((crop_label < 0)):
                        removed_cases += 1
                        
                    elif np.all((crop_label >= 0)): 
                        generated_cases += 1

                        
                        patch_mean       = np.mean(crop_label)
                        P_means.append(patch_mean)
                        
                        Block.append(block)
                        CID.append(block_id)
                        Cultivar.append(block_variety)
                        Trellis.append(block_trellis)
                        TID.append(block_tid)
                        RW.append(int(block_rw))
                        SP.append(int(block_sp))
                        YEAR.append(year)
                        X_COOR.append(i)
                        Y_COOR.append(j)
                                    

                        
        df['block']       = Block
        df['X']           = X_COOR
        df['Y']           = Y_COOR
        df['year']        = YEAR
        df['cultivar_id'] = CID
        df['cultivar']    = Cultivar
        df['trellis']     = Trellis
        df['trellis_id']  = TID
        df['row']         = RW
        df['space']       = SP
        df['patch_mean']     = P_means

        if self.cultivar_list is None:
            Newdf = df
        else: 
            Newdf = df[df['cultivar'].isin(self.cultivar_list)]

        return Newdf

    def split_data_(self): 

        full_dataframe  = self.hyper_df_csv()
        training_blocks_names, validation_blocks_names, testing_blocks_names = self.scenario3_split_data(full_dataframe)
        train = full_dataframe[full_dataframe['block'].isin(training_blocks_names)]
        val   = full_dataframe[full_dataframe['block'].isin(validation_blocks_names)]
        test  = full_dataframe[full_dataframe['block'].isin(testing_blocks_names)]

        return train, val, test

    def scenario3_split_data(self, df): 
        
        
        GroupList = df.groupby(by=["cultivar"]) 
        training_blocks_names, validation_blocks_names, testing_blocks_names = [], [], []

        for cul, frame in GroupList: 
            unique_blocks = frame['block'].unique()
            n_blocks      = len(frame['block'].unique())
                
            if n_blocks >= 3:
                if n_blocks == 3:
                    training_blocks_names.append(unique_blocks[0])
                    validation_blocks_names.append(unique_blocks[2]) 
                    testing_blocks_names.append(unique_blocks[1])

                elif n_blocks > 3:
                    blocks_names, blocks_means = [], []
                    for b in unique_blocks: 
                        blocks_names.append(b)
                        this_b_df = frame.loc[frame['block'] == b]
                        blocks_means.append(this_b_df['patch_mean'].mean())

                    # List of tuples with blocks and mean yield
                    block_mean_yield = [(blocks, mean) for blocks, mean in zip(blocks_names, blocks_means)]

                    block_mean_yield = sorted(block_mean_yield, key = lambda x: x[1], reverse = True)
                    # Print out the feature and importances 
                    #[print('Block: {:20} Mean: {}'.format(*pair)) for pair in block_mean_yield]

                    sorted_block_names = [block_mean_yield[i][0] for i in range(len(block_mean_yield))]
                    sorted_block_yield = [block_mean_yield[i][1] for i in range(len(block_mean_yield))]

                    te  = 1
                    val = 2
                    for i in range(len(block_mean_yield)):

                        names  = block_mean_yield[i][0]

                        if i == te: 
                            testing_blocks_names.append(names)
                            te = te + 3
                        elif i == val: 
                            validation_blocks_names.append(names)
                            val = val + 3
                        else:
                            training_blocks_names.append(names)

                
        return training_blocks_names, validation_blocks_names, testing_blocks_names

#-------------------------------------------------------------------------------------------#
#                                    Vineyard Data Vis                                      #
#-------------------------------------------------------------------------------------------#

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
    for i in range(30):
        if i == 0:
            Data  = df.loc[(df['ytrue'] >= i) & (df['ytrue'] <= (i+1))]
        else: 
            Data  = df.loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))] 
        counts.append(len(Data))
        MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
        if MAPE > 1: 
            MAPE = 1
        MAPE_Errors.append(MAPE*100)

    pearson_value = pearsonr(MAPE_Errors, counts)[0]
    bins_value  = np.arange(1, 31, 1)

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









blocks_information = {'LIV_003':['MALVASIA_BIANCA', '7', '12', '7', '1991', '4WIREWO', '1'], 
          'LIV_004':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_005':['CABERNET_SAUVIGNON', '2', '11', '5', '1996', 'LIVDC', '3'], 
          'LIV_006':['MALVASIA_BIANCA', '7', '12', '7', '1993', '4WIREWO', '1'], 
          'LIV_007':['SYMPHONY', '14', '10', '5', '1996', 'LIVDC', '3'], 
          'LIV_008':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_009':['PINOT_GRIS', '11', '10', '4', '2014', 'STACKEDT', '5'], 
          'LIV_010':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_011':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_012':['SYRAH', '15', '10', '8', '1995', 'LIVDC', '3'], 
          'LIV_013':['SYRAH', '15', '12', '7', '1995', 'LIVDC', '3'], 
          'LIV_014':['RIESLING', '13', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_015':['MALVASIA_BIANCA', '7', '12', '7', '1985', 'QUAD', '6'], 
          'LIV_016':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_017':['CABERNET_SAUVIGNON', '2', '11', '5', '1996', 'LIVDC', '3'], 
          'LIV_018':['CHARDONNAY', '3','10', '4', '1995', 'LIVDC', '3'], 
          'LIV_019':['RIESLING','13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_021':['PINOT_GRIS', '11', '10', '4', '2015', 'STACKEDT', '5'], 
          'LIV_022':['PINOT_NOIR', '12', '10', '5', '1997', 'SPLIT', '2'],
          'LIV_025':['CABERNET_SAUVIGNON', '2', '9', '9', '1996', '4WIREWO', '1'], 
          'LIV_026':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_027':['MERLOT', '8', '10', '4', '1994', 'LIVDC', '3'], 
          'LIV_028':['MUSCAT_CANELLI', '9', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_032':['CABERNET_SAUVIGNON', '2', '10', '5', '1996', 'LIVDC', '3'], 
          'LIV_038':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_050':['SYMPHONY', '14', '10', '7', '1997', 'LIVDC', '3'], 
          'LIV_058':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_061':['PINOT_GRIS', '11', '10', '6', '2002', '4WIREWO', '1'], 
          'LIV_062':['SYRAH', '15', '10', '8', '1995', 'LIVDC', '3'], 
          'LIV_063':['PINOT_GRIS', '11', '10', '4', '2014', 'STACKEDT', '5'], 
          'LIV_064':['CABERNET_SAUVIGNON', '2', '8', '9', '1997', '4WIREWO', '1'], 
          'LIV_066':['PINOT_GRIS', '11', '10', '4', '2012', 'SPLIT', '2'], 
          'LIV_068':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_070':['CABERNET_SAUVIGNON', '2', '8', '9', '1996', '4WIREWO', '1'], 
          'LIV_073':['PINOT_NOIR', '12', '12', '7', '2003', 'SPLIT', '2'], 
          'LIV_076':['RIESLING', '13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_077':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_089':['CHARDONNAY', '3', '9', '6', '2010', 'VERTICAL', '7'], 
          'LIV_090':['CHARDONNAY', '3', '11', '6', '1993', 'VERTICAL', '7'], 
          'LIV_094':['RIESLING', '13', '11', '5', '2014', 'STACKEDT', '5'], 
          'LIV_102':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_103':['CABERNET_SAUVIGNON', '2', '9', '6', '2012', 'HIGHWIRE', '8'], 
          'LIV_105':['MUSCAT_OF_ALEXANDRIA', '10', '10', '4', '2011', 'LIVDC', '3'], 
          'LIV_107':['RIESLING', '13', '11', '5', '2013', 'SPLIT', '2'], 
          'LIV_111':['CHARDONNAY', '3', '10', '4', '1995', 'LIVDC', '3'], 
          'LIV_114':['MALVASIA_BIANCA', '7', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_123':['MALBEC', '6', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_125':['DORNFELDER', '4', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_126':['CABERNET_SAUVIGNON', '2', '9', '6', '2010', 'TALLVERTICAL', '9'], 
          'LIV_128':['RIESLING', '13', '11', '5', '2013', 'SPLIT', '2'], 
          'LIV_135':['CHARDONNAY', '3', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_136':['ALICANTE_BOUSCHET', '1', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_163':['CHARDONNAY', '3', '12', '4', '1995', 'LIVDC', '3'], 
          'LIV_172':['MUSCAT_CANELLI', '9', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_175':['CHARDONNAY', '3', '11', '4', '1994', 'LIVDC', '3'], 
          'LIV_176':['CHARDONNAY', '3', '11', '7', '1994', '4WIREWO', '1'], 
          'LIV_177':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_178':['RIESLING', '13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_181':['SYMPHONY', '14', '10', '6', '1997', 'LIVDC', '3'], 
          'LIV_182':['LAMBRUSCO', '5', '11', '5', '2013', 'QUAD', '6'], 
          'LIV_186':['RIESLING', '13', '11', '7', '2012', 'SPLIT', '2'], 
          'LIV_193':['MERLOT', '8', '11', '5', '2010', 'SPLIT', '2']}
