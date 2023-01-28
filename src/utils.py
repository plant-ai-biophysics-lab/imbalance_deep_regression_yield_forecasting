import os, os.path
import pandas as pd
import numpy as np
np.random.seed(0)
import math
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import seaborn as sns
import cv2
import rasterio 
from rasterio.mask import mask
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
from rasterio.merge import merge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr

from src.configs import blocks_information



#-------------------------------------------------------------------------------------------#
#                                      Data Generation                                      #
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



def triple_emp_effective_weights_hist_plot(emp, effective, weights, method: str):
     
    fig, axs = plt.subplots(1, 3 , figsize = (21, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    bins_value  = np.arange(1, 31, 1)
    ax1 = sns.barplot(x = bins_value, y= emp, color = sns.color_palette()[0], width = 0.9, ax = axs[0])
    ax1.set_title("Emprical Label Distribution", fontsize = 14)


    ax2 = sns.barplot(x = bins_value, y= effective, color = sns.color_palette()[0], width = 0.9, ax = axs[1])
    ax2.set_title("Effective Label Distribution", fontsize = 14)

    ax3 = sns.barplot(x = bins_value, y= weights, color = sns.color_palette()[0], width = 0.9, ax = axs[2])
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

