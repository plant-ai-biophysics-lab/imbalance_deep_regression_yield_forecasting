import os
import os.path as path
import numpy as np
from glob import glob
import random
import pandas as pd
import torch
import torch.nn as nn
from KDEpy import FFTKDE
from sklearn.preprocessing import MinMaxScaler
from src.RWSampler import lds_prepare_weights, cb_prepare_weights, TargetRelevance

    

def getData(batch_size, lds_ks, lds_sigma, dw_alpha, betha, re_weighting_method: str, exp_name: str): 

    data_dir       = '/data2/hkaman/Livingston/data/10m/'
    botneck_size   = 2
    exp_output_dir = '/data2/hkaman/Imbalance/EXPs/' + 'EXP_' + exp_name

    
    isExist  = os.path.isdir(exp_output_dir)

    if not isExist:
        os.makedirs(exp_output_dir)
        os.makedirs(exp_output_dir + '/coords')
        os.makedirs(exp_output_dir + '/loss')

    train_csv = pd.read_csv('/data2/hkaman/Livingston/EXPs/10m/EXP_S3_UNetLSTM_10m_time/coords/train.csv', index_col=0)
    train_csv.to_csv(os.path.join(exp_output_dir + '/coords','train.csv'))
    valid_csv = pd.read_csv('/data2/hkaman/Livingston/EXPs/10m/EXP_S3_UNetLSTM_10m_time/coords/val.csv', index_col= 0)
    valid_csv.to_csv(os.path.join(exp_output_dir + '/coords','val.csv'))
    test_csv  = pd.read_csv('/data2/hkaman/Livingston/EXPs/10m/EXP_S3_UNetLSTM_10m_time/coords/test.csv', index_col= 0)
    test_csv.to_csv(os.path.join(exp_output_dir + '/coords','test.csv'))
    print(f"{train_csv.shape} | {valid_csv.shape} | {test_csv.shape}")
    #==============================================================================================================#
    #============================================ Imprical Data Weight Generation =================================#
    #==============================================================================================================#
    '''train_sampler, valid_sampler, test_sampler  = return_cost_sensitive_weight_sampler(train_csv, valid_csv, test_csv, exp_output_dir, run_status = 'train')

    train_weights = train_csv['NormWeight'].to_numpy() 
    train_weights = torch.DoubleTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, 
                                                                   len(train_weights), replacement=True)    

    val_weights   = valid_csv['NormWeight'].to_numpy() 
    val_weights   = torch.DoubleTensor(val_weights)
    val_sampler   = torch.utils.data.sampler.WeightedRandomSampler(val_weights, 
                                                                   len(val_weights), replacement=True)    

    test_weights = test_csv['NormWeight'].to_numpy() 
    test_weights = torch.DoubleTensor(test_weights)
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))'''

    #==============================================================================================================#
    #============================================     Reading Data                =================================#
    #==============================================================================================================#
    #csv_coord_dir = '/data2/hkaman/Livingston/EXPs/10m/EXP_S3_UNetLSTM_10m_time/'
    dataset_training = dataloader_RGB(data_dir, exp_output_dir, 
                                        category = 'train', 
                                        patch_size = 16, 
                                        in_channels = 6,
                                        lds_ks = lds_ks,
                                        lds_sigma = lds_sigma, 
                                        dw_alpha = dw_alpha, 
                                        betha = betha,
                                        re_weighting_method = re_weighting_method)

    dataset_validate = dataloader_RGB(data_dir, 
                                        exp_output_dir, 
                                        category = 'val',  
                                        patch_size = 16, 
                                        in_channels = 6,
                                        lds_ks = lds_ks,
                                        lds_sigma = lds_sigma,
                                        dw_alpha = dw_alpha, 
                                        betha = betha,
                                        re_weighting_method = re_weighting_method)
    

    dataset_test     = dataloader_RGB(data_dir, 
                                        exp_output_dir, 
                                        category = 'test',  
                                        patch_size = 16, 
                                        in_channels = 6,
                                        lds_ks = lds_ks,
                                        lds_sigma = lds_sigma,
                                        dw_alpha = dw_alpha, 
                                        betha = betha,
                                        re_weighting_method = re_weighting_method)     

    #==============================================================================================================#
    #=============================================      Data Loader               =================================#
    #==============================================================================================================#                      
    # define training and validation data loaders
    data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size= batch_size, 
                                                    shuffle=True, num_workers=8) #   sampler=train_sampler, 
    data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                    shuffle=False,  num_workers=8) #sampler=val_sampler,
    data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                    shuffle=False,  num_workers=8) #sampler=test_sampler,

    return data_loader_training, data_loader_validate, data_loader_test



class dataloader_RGB(object):
    def __init__(self, npy_dir, csv_dir, 
                                category: str, 
                                patch_size: int, 
                                in_channels: int, 
                                lds_ks: int, 
                                lds_sigma: int, 
                                dw_alpha: float,
                                betha: float,
                                re_weighting_method: str
                                ):

        self.npy_dir      = npy_dir
        self.csv_dir      = csv_dir
        self.wsize        = patch_size
        self.in_channels  = in_channels
        self.re_weighting_method = re_weighting_method


        if category    == 'train': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/train.csv', index_col=0) 
            self.NewDf.reset_index(inplace = True, drop = True)
        elif category  == 'val': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/val.csv', index_col=0)
            self.NewDf.reset_index(inplace = True, drop = True)
        elif category  == 'test': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/test.csv', index_col=0)
            self.NewDf.reset_index(inplace = True, drop = True)
        
        self.images = sorted(glob(os.path.join(self.npy_dir , 'imgs') +'/*.npy'))
        self.labels = sorted(glob(os.path.join(self.npy_dir , 'labels') +'/*.npy'))

        if re_weighting_method == 'lds':
            self.weights = self.return_pixelwise_weight_lds(lds_ks, lds_sigma)
        elif re_weighting_method == 'dw':
            self.weights = self.return_pixelwise_weight_dw(dw_alpha)
            #self.weights = np.where(self.weights >= 1, self.weights, 1)
        elif re_weighting_method == 'cb':
            self.weights = self.return_pixelwise_weight_cb(lds_ks, lds_sigma, betha)
            #self.weights = np.where(self.weights >= 1, self.weights, 1)


    def __getitem__(self, idx):

        xcoord      = self.NewDf.loc[idx]['X'] 
        ycoord      = self.NewDf.loc[idx]['Y'] 
        block_id    = self.NewDf.loc[idx]['block']
        cultivar    = self.NewDf.loc[idx]['cultivar']
        cultivar_id = self.NewDf.loc[idx]['cultivar_id']
        rw_id       = self.NewDf.loc[idx]['row']
        sp_id       = self.NewDf.loc[idx]['space']
        t_id        = self.NewDf.loc[idx]['trellis_id']
        
        WithinBlockMean    = self.NewDf.loc[idx]['win_block_mean']
        WithinBlockStd     = self.NewDf.loc[idx]['win_block_std']
        WithinCultivarMean = self.NewDf.loc[idx]['win_cultivar_mean']
        WithinCultivarStd  = self.NewDf.loc[idx]['win_cultivar_std']
        
        img_path   = self.NewDf.loc[idx]['IMG_PATH']
        label_path = self.NewDf.loc[idx]['LABEL_PATH']

        # return cropped input image using each patch coordinates
        image = self.crop_gen(img_path, xcoord, ycoord) 
        image = np.swapaxes(image, -1, 0)    
        if self.in_channels == 5: 
            bc_mean_mtx = self.add_input_within_bc_mean(WithinBlockMean)
            image = np.concatenate([image, bc_mean_mtx], axis = 0)

        if self.in_channels == 6: 
            bc_mean_mtx = self.add_input_within_bc_mean(WithinBlockMean)
            block_timeseries_encode = self.time_series_encoding(block_id)
            image = np.concatenate([image, bc_mean_mtx, block_timeseries_encode], axis = 0)
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image / 255.

        # return embedding tensor: 
        CulMatrix = self.patch_cultivar_matrix(cultivar_id)  
        RWMatrix  = self.patch_rw_matrix(rw_id) 
        SpMatrix  = self.patch_sp_matrix(sp_id)
        TMatrix   = self.patch_tid_matrix(t_id)  
        EmbMat    = np.concatenate([CulMatrix, RWMatrix, SpMatrix, TMatrix], axis = 0)

        # return crooped mask tensor: 
        mask  = self.crop_gen(label_path, xcoord, ycoord) 
        mask  = np.swapaxes(mask, -1, 0)
        mask  = torch.as_tensor(mask)

        weight_mtx = self.weights[idx, :, :]
        weight_mtx = np.expand_dims(weight_mtx, axis = 0)
        weight_mtx = torch.as_tensor(weight_mtx)


        sample = {"image": image, "mask": mask, "EmbMatrix": EmbMat, "block": block_id, "cultivar": cultivar, 
                "X": xcoord, "Y": ycoord, "win_block_mean":WithinBlockMean,  "weight": weight_mtx}
        
        return sample

    def __len__(self):
        return len(self.NewDf)
    
    def return_pixelwise_weight_lds(self, lds_ks, lds_sigma):

        masks = None
        for idx, row in self.NewDf.iterrows():
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
        weights = lds_prepare_weights(reshaped_masks, 'inverse', 
                                    max_target=30, 
                                    lds=True, 
                                    lds_kernel='gaussian', 
                                    lds_ks   =lds_ks, 
                                    lds_sigma=lds_sigma)
        
        weights = np.reshape(weights, (masks.shape[0], masks.shape[1], masks.shape[2]))


        return weights
    def return_pixelwise_weight_cb(self, lds_ks, lds_sigma, betha):

        masks = None
        for idx, row in self.NewDf.iterrows():
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
        weights =  cb_prepare_weights(reshaped_masks, lds_kernel='gaussian', lds_ks=lds_ks, lds_sigma=lds_sigma, betha=betha)

        weights = np.reshape(weights, (masks.shape[0], masks.shape[1], masks.shape[2]))
        return weights  
    def return_pixelwise_weight_dw(self, dw_alpha):

        masks = None
        for idx, row in self.NewDf.iterrows():
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

        #weights = TargetRelevance(reshaped_masks, alpha = dw_alpha).get_relevance()
        weights = TargetRelevance(reshaped_masks, alpha = dw_alpha).__call__(reshaped_masks)

        weights = np.reshape(weights, (masks.shape[0], masks.shape[1], masks.shape[2]))
        return weights
    

    def crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        crop_src = src[:, xcoord:xcoord + self.wsize, ycoord:ycoord + self.wsize, :]
        return crop_src 
  
    def patch_cultivar_matrix(self, cul_id):
        zeros_matrix       = np.full(4, (1/cul_id))
        cultivar_matrix    = zeros_matrix.reshape(1, int(self.wsize/8), int(self.wsize/8))
        
        return cultivar_matrix
    
    def patch_rw_matrix(self, rw):
        zeros_matrix       = np.full(4, (1/rw))
        rw_matrix          = zeros_matrix.reshape(1,int(self.wsize/8), int(self.wsize/8))
        
        return rw_matrix    
    
    def patch_sp_matrix(self, sp):
        zeros_matrix       = np.full(4, (1/sp))
        sp_matrix          = zeros_matrix.reshape(1,int(self.wsize/8), int(self.wsize/8))
        
        return sp_matrix
        
    def patch_tid_matrix(self, tid):
        zeros_matrix = np.full(4, (1/tid))
        tid_matrix = zeros_matrix.reshape(1,int(self.wsize/8), int(self.wsize/8))
        
        return tid_matrix
    
    def add_input_within_bc_mean(self, bloks_mean):
        fill_matrix_bmean = np.full((1, self.wsize, self.wsize, 15), bloks_mean) 
        
        return fill_matrix_bmean
    
    def time_series_encoding(self, block_id):
        timeseries = None

        name_split = os.path.split(str(block_id))[-1]
        year       = name_split[-4:]

        if year == '2016': 
            days = [91, 95, 107, 116, 135, 136, 141, 150, 161, 166, 171, 176, 182, 195, 202]
        elif year =='2017':
            days = [90, 109, 119, 134, 140, 156, 169, 176, 179, 184, 190, 192, 195, 197, 202]
        elif year =='2018':
            days = [91, 105, 112, 115, 121, 131, 135, 142, 152, 155, 165, 175, 185, 191, 194]
        elif year =='2019':
            days = [91, 101, 112, 115, 121, 124, 131, 145, 152, 155, 164, 171, 181, 192, 202]
        
        for day in days: 
            this_week_matrix = np.full((1, self.wsize, self.wsize), 1 - np.sin(day/(365*np.pi))) 
            this_week_matrix = np.expand_dims(this_week_matrix, axis = -1)
            if timeseries is None:
                timeseries = this_week_matrix
            else:
                timeseries   = np.concatenate([timeseries, this_week_matrix], axis = -1)

        return timeseries  
    