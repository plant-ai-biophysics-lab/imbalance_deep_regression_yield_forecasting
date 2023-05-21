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
from src.RWSampler import lds_prepare_weights, cb_prepare_weights, TargetRelevance, return_cost_sensitive_weight_sampler
from src import configs
    

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

    '''train_csv, valid_csv, test_csv = data_generator(eval_scenario = 'block_year_hold_out', 
                                                    spatial_resolution =10, 
                                                    patch_size = 16, 
                                                    patch_offset =2,  
                                                    cultivar_list = None, 
                                                    year_list = ['2016', '2019', '2018', '2017']).return_split_dataframe()
    train_csv.to_csv(os.path.join(exp_output_dir + '/coords','train.csv'))
    valid_csv.to_csv(os.path.join(exp_output_dir + '/coords','val.csv'))
    test_csv.to_csv(os.path.join(exp_output_dir + '/coords','test.csv'))
    print(f"{train_csv.shape} | {valid_csv.shape} | {test_csv.shape}")'''
    #==============================================================================================================#
    #============================================ Imprical Data Weight Generation =================================#
    #==============================================================================================================#
    
    #train_sampler, valid_sampler, test_sampler  = return_cost_sensitive_weight_sampler(train_csv, valid_csv, test_csv, exp_output_dir, run_status = 'train')

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
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights), replacement=True)
    #==============================================================================================================#
    #============================================     Reading Data                =================================#
    #==============================================================================================================#
    #csv_coord_dir = '/data2/hkaman/Livingston/EXPs/10m/EXP_S3_UNetLSTM_10m_time/'
    dataset_training = dataloader_RGB(data_dir, exp_output_dir, 
                                        category = 'train', 
                                        patch_size = 16, 
                                        in_channels = 5,
                                        lds_ks = lds_ks,
                                        lds_sigma = lds_sigma, 
                                        dw_alpha = dw_alpha, 
                                        betha = betha,
                                        re_weighting_method = re_weighting_method)

    dataset_validate = dataloader_RGB(data_dir, 
                                        exp_output_dir, 
                                        category = 'val',  
                                        patch_size = 16, 
                                        in_channels = 5,
                                        lds_ks = lds_ks,
                                        lds_sigma = lds_sigma,
                                        dw_alpha = dw_alpha, 
                                        betha = betha,
                                        re_weighting_method = re_weighting_method)
    

    dataset_test     = dataloader_RGB(data_dir, 
                                        exp_output_dir, 
                                        category = 'test',  
                                        patch_size = 16, 
                                        in_channels = 5,
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
                                                    shuffle=True,    num_workers=8) #  sampler=train_sampler,
    data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                    shuffle=False, num_workers=8) # sampler=val_sampler, 
    data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                    shuffle=False,   num_workers=8) # sampler=test_sampler,

    return data_loader_training, data_loader_validate, data_loader_test


class data_generator():
    def __init__(self, eval_scenario: str, 
                    spatial_resolution: int, 
                    patch_size: int, 
                    patch_offset: int,  
                    cultivar_list: list, 
                    year_list: list):

        self.eval_scenario      = eval_scenario
        self.spatial_resolution = spatial_resolution
        self.patch_size         = patch_size
        self.patch_offset       = patch_offset
        self.cultivar_list      = cultivar_list
        self.year_list          = year_list


        if self.cultivar_list is None: 
            self.cultivar_list = ['MALVASIA_BIANCA', 'MUSCAT_OF_ALEXANDRIA', 
                                    'CABERNET_SAUVIGNON','SYMPHONY', 
                                    'MERLOT', 'CHARDONNAY', 
                                    'SYRAH', 'RIESLING']

        if self.spatial_resolution == 1: 
            self.npy_dir = '/data2/hkaman/Livingston/data/1m/'

        else: 
            self.npy_dir = '/data2/hkaman/Livingston/data/10m/'

        self.images_dir  = os.path.join(self.npy_dir, 'imgs')
        self.image_names = os.listdir(self.images_dir)
        self.image_names.sort() 

        self.label_dir   = os.path.join(self.npy_dir, 'labels')
        self.label_names = os.listdir(self.label_dir)
        self.label_names.sort() 


    def return_split_dataframe(self):

        full_dataframe = self.return_dataframe_patch_info()


        if self.eval_scenario == 'pixel_hold_out': 
            train, valid, test = self.pixel_hold_out(full_dataframe)
        elif self.eval_scenario == 'year_hold_out':
            train, valid, test = self.year_hold_out(full_dataframe)
        elif self.eval_scenario == 'block_hold_out': 
            train, valid, test = self.block_hold_out(full_dataframe)
        elif self.eval_scenario == 'block_year_hold_out': 
            train, valid, test = self.block_year_hold_out(full_dataframe)

        '''print(f"Training Patches: {len(train)}, Validation: {len(valid)} and Test: {len(test)}")
        print("============================= Train =========================================")
        _ = print_df_summary(train)
        print("============================= Validation ====================================")
        _ = print_df_summary(valid)
        print("============================= Test ==========================================")
        _ = print_df_summary(test)
        print("=============================================================================")'''

        return train, valid, test


    def return_dataframe_patch_info(self): 

        df = pd.DataFrame()

        Block, Cultivar, CID, Trellis, TID, RW, SP = [], [], [], [], [], [], []
        P_means, YEAR, X_COOR, Y_COOR, IMG_P, Label_P  = [], [], [], [], [], []
        
        
        generated_cases = 0
        removed_cases = 0 
        
        
        for idx, name in enumerate(self.label_names):
            # Extract Image path
            image_path = os.path.join(self.images_dir, self.image_names[idx])

            name_split  = os.path.split(name)[-1]
            block_name  = name_split.replace(name_split[12:], '')
            root_name   = name_split.replace(name_split[7:], '')
            year        = name_split.replace(name_split[0:8], '').replace(name_split[12:], '')
            
            res           = {key: configs.blocks_information[key] for key in configs.blocks_information.keys() & {root_name}}
            list_d        = res.get(root_name)
            block_variety = list_d[0]
            block_id      = list_d[1]
            block_rw      = list_d[2]
            block_sp      = list_d[3]
            block_trellis = list_d[5]
            block_tid     = list_d[6]

            label_npy = os.path.join(self.label_dir, name)
            label = np.load(label_npy, allow_pickle=True)
            label = label[0,:,:,0]
            width, height = label.shape[1], label.shape[0]
            
            
            for i in range(0, height - self.patch_size, self.patch_offset):
                for j in range(0, width - self.patch_size, self.patch_offset):
                    crop_label = label[i:i+ self.patch_size, j:j+ self.patch_size]
                    
                    if np.any((crop_label < 0)):
                        removed_cases += 1
                        
                    elif np.all((crop_label >= 0)): 

                        generated_cases += 1
                        
                        patch_mean       = np.mean(crop_label)
                        P_means.append(patch_mean)
                    
                        Block.append(block_name)
                        CID.append(block_id)
                        Cultivar.append(block_variety)
                        Trellis.append(block_trellis)
                        TID.append(block_tid)
                        RW.append(int(block_rw))
                        SP.append(int(block_sp))
                        YEAR.append(year)
                        X_COOR.append(i)
                        Y_COOR.append(j)
                        IMG_P.append(image_path)
                        Label_P.append(label_npy)                                        

                        
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
        df['IMG_PATH']    = IMG_P
        df['LABEL_PATH']  = Label_P
        
        if self.cultivar_list is None:
            Customized_df = df
            
        else: 
            Customized_df = df[df['cultivar'].isin(self.cultivar_list)]
            
        return Customized_df


    def year_hold_out(self, df): 

        NewGroupedDf = df.groupby(by=["year"])

        Group1 = NewGroupedDf.get_group(self.year_list[0])
        Group2 = NewGroupedDf.get_group(self.year_list[1])
        Group3 = NewGroupedDf.get_group(self.year_list[2])
        Group4 = NewGroupedDf.get_group(self.year_list[3])

        frames = [Group1, Group2]
        train = pd.concat(frames)
        valid = Group3
        test  = Group4

        return train, valid, test
    
    def block_hold_out(self, df):
        
        datafram_grouby_year = df.groupby(by = 'year')
        dataframe_year2017   = datafram_grouby_year.get_group('2017')
        
        new_dataframe_basedon_block_mean = pd.DataFrame()
        block_root_name, cultivar, b_mean = [], [], []
        
        dataframe_year2017_groupby_block = dataframe_year2017.groupby(by = 'block')

        for block, blockdf in dataframe_year2017_groupby_block:
            name_split = os.path.split(block)[-1]
            root_name  = name_split.replace(name_split[7:], '')
            block_root_name.append(root_name)
            
            cultivar.append(blockdf['cultivar'].iloc[0])
            b_mean.append(blockdf['patch_mean'].mean())
            
        new_dataframe_basedon_block_mean['block'] = block_root_name
        new_dataframe_basedon_block_mean['cultivar'] = cultivar
        new_dataframe_basedon_block_mean['block_mean'] = b_mean
            
        # split sorted blocks and then split within each cultivar 
        BlockMeanBased_GroupBy_Cultivar = new_dataframe_basedon_block_mean.groupby(by=["cultivar"]) 
        training_blocks_names = []
        validation_blocks_names = []
        testing_blocks_names = []
        
        for cul, frame in BlockMeanBased_GroupBy_Cultivar: 
            n_blocks = len(frame.loc[frame['cultivar'] == cul])
            
            if n_blocks <= 1: 
                name_2016  = frame['block'].iloc[0] + '_2016'
                name_2017  = frame['block'].iloc[0] + '_2017'
                name_2018  = frame['block'].iloc[0] + '_2018'
                name_2019  = frame['block'].iloc[0] + '_2019'
                training_blocks_names.extend((name_2016, name_2017, name_2018, name_2019))
                
            elif n_blocks == 2:
                name_2016_0  = frame['block'].iloc[0] + '_2016'
                name_2017_0  = frame['block'].iloc[0] + '_2017'
                name_2018_0  = frame['block'].iloc[0] + '_2018'
                name_2019_0  = frame['block'].iloc[0] + '_2019'
                
                training_blocks_names.extend((name_2016_0, name_2017_0, name_2018_0, name_2019_0))
                
                name_2016_1  = frame['block'].iloc[1] + '_2016'
                name_2017_1  = frame['block'].iloc[1] + '_2017'
                name_2018_1  = frame['block'].iloc[1] + '_2018'
                name_2019_1  = frame['block'].iloc[1] + '_2019'
                
                validation_blocks_names.extend((name_2016_1, name_2017_1, name_2018_1, name_2019_1))
                
            elif n_blocks == 3:
                name_2016_0  = frame['block'].iloc[0] + '_2016'
                name_2017_0  = frame['block'].iloc[0] + '_2017'
                name_2018_0  = frame['block'].iloc[0] + '_2018'
                name_2019_0  = frame['block'].iloc[0] + '_2019'
                
                training_blocks_names.extend((name_2016_0, name_2017_0, name_2018_0, name_2019_0))
                
                name_2016_1  = frame['block'].iloc[2] + '_2016'
                name_2017_1  = frame['block'].iloc[2] + '_2017'
                name_2018_1  = frame['block'].iloc[2] + '_2018'
                name_2019_1  = frame['block'].iloc[2] + '_2019'
                
                testing_blocks_names.extend((name_2016_1, name_2017_1, name_2018_1, name_2019_1))  
                
                name_2016_2  = frame['block'].iloc[1] + '_2016'
                name_2017_2  = frame['block'].iloc[1] + '_2017'
                name_2018_2  = frame['block'].iloc[1] + '_2018'
                name_2019_2  = frame['block'].iloc[1] + '_2019'
                
                validation_blocks_names.extend((name_2016_2, name_2017_2, name_2018_2, name_2019_2)) 
                
            elif n_blocks > 3:
                blocks_2017      = frame['block']
                blocks_mean_2017 = frame['block_mean']

                # List of tuples with blocks and mean yield
                block_mean_yield_2017 = [(blocks, mean) for blocks, 
                                    mean in zip(blocks_2017, blocks_mean_2017)]

                block_mean_yield_2017 = sorted(block_mean_yield_2017, key = lambda x: x[1], reverse = True)
 

                te  = 1
                val = 2
                for i in range(len(block_mean_yield_2017)):
                    name_2016  = block_mean_yield_2017[i][0] + '_2016'
                    name_2017  = block_mean_yield_2017[i][0] + '_2017'
                    name_2018  = block_mean_yield_2017[i][0] + '_2018'
                    name_2019  = block_mean_yield_2017[i][0] + '_2019'

                    if i == te: 
                        testing_blocks_names.append(name_2016)
                        testing_blocks_names.append(name_2017)
                        testing_blocks_names.append(name_2018)
                        testing_blocks_names.append(name_2019)
                        te = te + 3
                    elif i == val: 
                        validation_blocks_names.append(name_2016)
                        validation_blocks_names.append(name_2017)
                        validation_blocks_names.append(name_2018)
                        validation_blocks_names.append(name_2019)

                        val = val + 3
                    else:
                        training_blocks_names.append(name_2016)
                        training_blocks_names.append(name_2017)
                        training_blocks_names.append(name_2018)
                        training_blocks_names.append(name_2019)

        train = df[df['block'].isin(training_blocks_names)]
        valid = df[df['block'].isin(validation_blocks_names)]
        test  = df[df['block'].isin(testing_blocks_names)] 


        return train, valid, test

    def block_year_hold_out(self, df):
        
        datafram_grouby_year = df.groupby(by = 'year')
        dataframe_year2017   = datafram_grouby_year.get_group('2017')
        
        new_dataframe_basedon_block_mean = pd.DataFrame()
        block_root_name, cultivar, b_mean = [], [], []
        
        dataframe_year2017_groupby_block = dataframe_year2017.groupby(by = 'block')

        for block, blockdf in dataframe_year2017_groupby_block:
            name_split = os.path.split(block)[-1]
            root_name  = name_split.replace(name_split[7:], '')
            block_root_name.append(root_name)
            
            cultivar.append(blockdf['cultivar'].iloc[0])
            b_mean.append(blockdf['patch_mean'].mean())
            
        new_dataframe_basedon_block_mean['block']      = block_root_name
        new_dataframe_basedon_block_mean['cultivar']   = cultivar
        new_dataframe_basedon_block_mean['block_mean'] = b_mean
        
        # split sorted blocks and then split within each cultivar 
        BlockMeanBased_GroupBy_Cultivar = new_dataframe_basedon_block_mean.groupby(by=["cultivar"]) 

        training_blocks_names = []
        validation_blocks_names = []
        testing_blocks_names = []
        
        for cul, frame in BlockMeanBased_GroupBy_Cultivar: 

            n_blocks = len(frame.loc[frame['cultivar'] == cul])
            
            if frame.shape[0] == 3:
                

                name_0  = frame['block'].iloc[0] + '_' + self.year_list[0]
                name_1  = frame['block'].iloc[0] + '_' + self.year_list[1]
                training_blocks_names.append(name_0)
                training_blocks_names.append(name_1)
                
                name_2  = frame['block'].iloc[1] + '_' + self.year_list[2]
                validation_blocks_names.append(name_2) 

                name_3  = frame['block'].iloc[2] + '_' + self.year_list[3]
                testing_blocks_names.append(name_3) 


                
            elif frame.shape[0] > 3:

                blocks_2017      = frame['block']
                blocks_mean_2017 = frame['block_mean']

                # List of tuples with blocks and mean yield
                block_mean_yield_2017 = [(blocks, mean) for blocks, 
                                    mean in zip(blocks_2017, blocks_mean_2017)]

                block_mean_yield_2017 = sorted(block_mean_yield_2017, key = lambda x: x[1], reverse = True)
                #print(block_mean_yield_2017)
                #print("============================")

                te  = 1
                val = 2
                for i in range(len(block_mean_yield_2017)):

                    if i == te: 
                        name_3  = block_mean_yield_2017[i][0] + '_' + self.year_list[3]
                        testing_blocks_names.append(name_3)
                        te = te + 3
                        #print(f"{cul}: {name_3}")
                    elif i == val: 
                        name_2  = block_mean_yield_2017[i][0] + '_' + self.year_list[2]
                        validation_blocks_names.append(name_2)
                        val = val + 3
                        #print(f"{cul}: {name_2}")
                    else:
                        name_0  = block_mean_yield_2017[i][0] + '_' + self.year_list[0]
                        name_1  = block_mean_yield_2017[i][0] + '_' + self.year_list[1]
                        #print(f"{cul}: {name_0, name_1}")
                        training_blocks_names.append(name_0)
                        training_blocks_names.append(name_1)
                    #print(f"with MORE than 3: {name_0, name_1, name_2, name_3}")

        #print(validation_blocks_names)
        train = df[df['block'].isin(training_blocks_names)]
        valid = df[df['block'].isin(validation_blocks_names)]
        test  = df[df['block'].isin(testing_blocks_names)] 


        return train, valid, test


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
            assert not np.isnan(self.weights).any()
            #self.weights = self.weights / np.max(self.weights)
            self.weights = np.where(self.weights >= 1, self.weights, 1)
            #print(f"Min weight {np.min(self.weights)} | Max weight {np.max(self.weights)}")
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
        
        #WithinBlockMean    = self.NewDf.loc[idx]['win_block_mean']
        #WithinBlockStd     = self.NewDf.loc[idx]['win_block_std']
        #WithinCultivarMean = self.NewDf.loc[idx]['win_cultivar_mean']
        #WithinCultivarStd  = self.NewDf.loc[idx]['win_cultivar_std']
        
        img_path   = self.NewDf.loc[idx]['IMG_PATH']
        label_path = self.NewDf.loc[idx]['LABEL_PATH']

        # return cropped input image using each patch coordinates
        image = self.crop_gen(img_path, xcoord, ycoord) 
        image = np.swapaxes(image, -1, 0)    
        if self.in_channels == 5: 
            block_timeseries_encode = self.time_series_encoding(block_id)
            image = np.concatenate([image, block_timeseries_encode], axis = 0)


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
        mask  = torch.as_tensor(mask, dtype=torch.float32)

        weight_mtx = self.weights[idx, :, :]
        weight_mtx = np.expand_dims(weight_mtx, axis = 0)
        weight_mtx = torch.as_tensor(weight_mtx, dtype=torch.float32)
        #weight_mtx = weight_mtx/torch.max(weight_mtx)

        #mean_w     = torch.diagonal(weight_mtx[0, :, :], 0)
        sample = {"image": image, "mask": mask, "EmbMatrix": EmbMat, "block": block_id, "cultivar": cultivar, 
                "X": xcoord, "Y": ycoord, "weight": weight_mtx} #, 'batch_w' : mean_w
        
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
    