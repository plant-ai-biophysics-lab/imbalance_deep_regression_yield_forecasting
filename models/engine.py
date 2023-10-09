import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geomloss import SamplesLoss   # ImagesLoss
device = "cuda" if torch.cuda.is_available() else "cpu"

from utils.losses import *
from models import configs


#======================================================================================================================================#
#====================================================== Training Config ===============================================================#
#======================================================================================================================================#   

class EarlyStopping():
    def __init__(self, tolerance=30, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, status):

        if status is True:
            self.counter = 0
        elif status is False: 
            self.counter +=1

        print(f"count: {self.counter}")
        if self.counter >= self.tolerance:  
                self.early_stop = True

def save_loss_df(loss_stat, loss_df_name, loss_fig_name):

    df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    df.to_csv(loss_df_name) 
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.ylim(0, df['value'].max())
    plt.savefig(loss_fig_name, dpi = 300)

def predict(model, data_loader_training, data_loader_validate, data_loader_testing, Exp_name = None,): 

    exp_output_dir = '/data2/hkaman/Imbalance/EXPs/' + 'EXP_' + Exp_name

    best_model_name      = exp_output_dir + '/best_model_' + Exp_name + '.pth'
    train_df_name        = exp_output_dir + '/' + Exp_name + '_train.csv'
    valid_df_name        = exp_output_dir + '/' + Exp_name + '_valid.csv'
    test_df_name         = exp_output_dir + '/' + Exp_name + '_test.csv'
    timeseries_fig       = exp_output_dir + '/' + Exp_name + '_timeseries.png'
    scatterplot          = exp_output_dir + '/' + Exp_name + '_scatterplot.png'

    train_bc_df1_name    = exp_output_dir + '/' + Exp_name + '_train_bc1.csv'
    train_bc_df2_name    = exp_output_dir + '/' + Exp_name + '_train_bc2.csv'

    valid_bc_df1_name    = exp_output_dir + '/' + Exp_name + '_valid_bc1.csv'
    valid_bc_df2_name    = exp_output_dir + '/' + Exp_name + '_valid_bc2.csv'

    test_bc_df1_name      = exp_output_dir + '/' + Exp_name + '_test_vis_bc1.csv'
    test_bc_df2_name     = exp_output_dir + '/' + Exp_name + '_test_vis_bc2.csv'

    model.load_state_dict(torch.load(best_model_name))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_output_files = []
    valid_output_files = []
    test_output_files = []

    with torch.no_grad():
        
        model.eval()
        #================ Train===========================
        # for batch, sample in enumerate(data_loader_training):
            
        #     X_batch_train       = sample['image'].to(device)
        #     y_batch_train       = sample['mask'].to(device)
        #     C_batch_train       = sample['EmbMatrix'].to(device)
        #     ID_batch_train      = sample['block']
        #     Cult_batch_train    = sample['cultivar']
        #     Xcoord_batch_train  = sample['X']
        #     ycoord_batch_train  = sample['Y']
        
            
        #     list_y_train_pred = model(X_batch_train, C_batch_train)
            
        #     y_true_train = y_batch_train.detach().cpu().numpy()
            
        #     ytpw1 = list_y_train_pred[0].detach().cpu().numpy()
        #     ytpw2 = list_y_train_pred[1].detach().cpu().numpy()
        #     ytpw3 = list_y_train_pred[2].detach().cpu().numpy()
        #     ytpw4 = list_y_train_pred[3].detach().cpu().numpy()
        #     ytpw5 = list_y_train_pred[4].detach().cpu().numpy()
        #     ytpw6 = list_y_train_pred[5].detach().cpu().numpy()
        #     ytpw7 = list_y_train_pred[6].detach().cpu().numpy()
        #     ytpw8 = list_y_train_pred[7].detach().cpu().numpy()
        #     ytpw9 = list_y_train_pred[8].detach().cpu().numpy()
        #     ytpw10 = list_y_train_pred[9].detach().cpu().numpy()
        #     ytpw11 = list_y_train_pred[10].detach().cpu().numpy()
        #     ytpw12 = list_y_train_pred[11].detach().cpu().numpy()
        #     ytpw13 = list_y_train_pred[12].detach().cpu().numpy()
        #     ytpw14 = list_y_train_pred[13].detach().cpu().numpy()
        #     ytpw15 = list_y_train_pred[14].detach().cpu().numpy()


        #     this_batch_train = {"block": ID_batch_train, "cultivar": Cult_batch_train, "X": Xcoord_batch_train, "Y": ycoord_batch_train,
        #                         "ytrue": y_true_train, "ypred_w1": ytpw1, "ypred_w2": ytpw2,"ypred_w3": ytpw3,"ypred_w4": ytpw4,"ypred_w5": ytpw5,"ypred_w6": ytpw6,"ypred_w7": ytpw7,"ypred_w8": ytpw8,
        #                         "ypred_w9": ytpw9,"ypred_w10": ytpw10,"ypred_w11": ytpw11,"ypred_w12": ytpw12,"ypred_w13": ytpw13,"ypred_w14": ytpw14,"ypred_w15": ytpw15}
            
        #     train_output_files.append(this_batch_train)

        # train_df = Inference.ScenarioEvaluation2D(train_output_files)
        # train_df.to_csv(train_df_name)
        # print("train evaluation is done!")
        # #train_block_names      = utils.npy_block_names(train_output_files)
        # #df1d_train, df2d_train = utils.time_series_eval_csv(train_output_files, train_block_names, patch_size)
        # #df1d_train.to_csv(train_bc_df1_name)
        # #df2d_train.to_csv(train_bc_df2_name)

        # #================== Validaiton====================
        # for batch, sample in enumerate(data_loader_validate):
            
        #     X_batch_val       = sample['image'].to(device)
        #     y_batch_val       = sample['mask'].to(device)
        #     C_batch_val       = sample['EmbMatrix'].to(device)
        #     ID_batch_val      = sample['block']
        #     Cult_batch_val    = sample['cultivar']
        #     Xcoord_batch_val  = sample['X']
        #     ycoord_batch_val  = sample['Y']


        #     list_y_val_pred = model(X_batch_val, C_batch_val)
                
        #     y_true_val    = y_batch_val.detach().cpu().numpy()

        #     yvpw1 = list_y_val_pred[0].detach().cpu().numpy()
        #     yvpw2 = list_y_val_pred[1].detach().cpu().numpy()
        #     yvpw3 = list_y_val_pred[2].detach().cpu().numpy()
        #     yvpw4 = list_y_val_pred[3].detach().cpu().numpy()
        #     yvpw5 = list_y_val_pred[4].detach().cpu().numpy()
        #     yvpw6 = list_y_val_pred[5].detach().cpu().numpy()
        #     yvpw7 = list_y_val_pred[6].detach().cpu().numpy()
        #     yvpw8 = list_y_val_pred[7].detach().cpu().numpy()
        #     yvpw9 = list_y_val_pred[8].detach().cpu().numpy()
        #     yvpw10 = list_y_val_pred[9].detach().cpu().numpy()
        #     yvpw11 = list_y_val_pred[10].detach().cpu().numpy()
        #     yvpw12 = list_y_val_pred[11].detach().cpu().numpy()
        #     yvpw13 = list_y_val_pred[12].detach().cpu().numpy()
        #     yvpw14 = list_y_val_pred[13].detach().cpu().numpy()
        #     yvpw15 = list_y_val_pred[14].detach().cpu().numpy()
            

        #     this_batch_val = {"block": ID_batch_val, "cultivar": Cult_batch_val, "X": Xcoord_batch_val, "Y": ycoord_batch_val, "ytrue": y_true_val, "ypred_w1": yvpw1, "ypred_w2": yvpw2, "ypred_w3": yvpw3, "ypred_w4": yvpw4, "ypred_w5": yvpw5, "ypred_w6": yvpw6, "ypred_w7": yvpw7, "ypred_w8": yvpw8,
        #                         "ypred_w9": yvpw9, "ypred_w10": yvpw10, "ypred_w11": yvpw11, "ypred_w12": yvpw12, "ypred_w13": yvpw13, "ypred_w14": yvpw14, "ypred_w15": yvpw15} 

                
        #     valid_output_files.append(this_batch_val)
        # # save the prediction in data2 drectory as a npy file
        # #np.save(valid_npy_name, valid_output_files)
        # valid_df = Inference.ScenarioEvaluation2D(valid_output_files)
        # valid_df.to_csv(valid_df_name)

        # #valid_block_names  = utils.npy_block_names(valid_output_files)
        # #df1d_valid, df2d_valid = utils.time_series_eval_csv(valid_output_files, valid_block_names, patch_size)
        # #df1d_valid.to_csv(valid_bc_df1_name)
        # #df2d_valid.to_csv(valid_bc_df2_name)
        # print("validation evaluation is done!")
        #=================== Test ========================
        for batch, sample in enumerate(data_loader_testing):
            
            X_batch_test       = sample['image'].to(device)
            y_batch_test       = sample['mask'].to(device)
            C_batch_test       = sample['EmbMatrix'].to(device)
            ID_batch_test      = sample['block']
            Cult_batch_test    = sample['cultivar']
            Xcoord_batch_test  = sample['X']
            ycoord_batch_test  = sample['Y']


            list_y_test_pred = model(X_batch_test, C_batch_test)
            y_true_test = y_batch_test.detach().cpu().numpy()
            
            ytepw1 = list_y_test_pred[0].detach().cpu().numpy()
            ytepw2 = list_y_test_pred[1].detach().cpu().numpy()
            ytepw3 = list_y_test_pred[2].detach().cpu().numpy()
            ytepw4 = list_y_test_pred[3].detach().cpu().numpy()
            ytepw5 = list_y_test_pred[4].detach().cpu().numpy()
            ytepw6 = list_y_test_pred[5].detach().cpu().numpy()
            ytepw7 = list_y_test_pred[6].detach().cpu().numpy()
            ytepw8 = list_y_test_pred[7].detach().cpu().numpy()
            ytepw9 = list_y_test_pred[8].detach().cpu().numpy()
            ytepw10 = list_y_test_pred[9].detach().cpu().numpy()
            ytepw11 = list_y_test_pred[10].detach().cpu().numpy()
            ytepw12 = list_y_test_pred[11].detach().cpu().numpy()
            ytepw13 = list_y_test_pred[12].detach().cpu().numpy()
            ytepw14 = list_y_test_pred[13].detach().cpu().numpy()
            ytepw15 = list_y_test_pred[14].detach().cpu().numpy()

            this_batch_test = {"block": ID_batch_test, "cultivar": Cult_batch_test, "X": Xcoord_batch_test, "Y": ycoord_batch_test, 
                            "ytrue": y_true_test, "ypred_w1": ytepw1, "ypred_w2": ytepw2, "ypred_w3": ytepw3, "ypred_w4": ytepw4, "ypred_w5": ytepw5, "ypred_w6": ytepw6, "ypred_w7": ytepw7, 
                            "ypred_w8": ytepw8, "ypred_w9": ytepw9, "ypred_w10": ytepw10, "ypred_w11": ytepw11, "ypred_w12": ytepw12, "ypred_w13": ytepw13, "ypred_w14": ytepw14, "ypred_w15": ytepw15}
            
            
            test_output_files.append(this_batch_test)
        #np.save(test_npy_name, test_output_files) 
        #print("Test Data is Saved!")
        test_df = Inference.ScenarioEvaluation2D(test_output_files)
        test_df.to_csv(test_df_name)
        print("test evaluation is done!")
        test_block_names  = ['LIV_186_2017', 'LIV_025_2019','LIV_105_2018', 'LIV_105_2019', 'LIV_181_2017', 'LIV_181_2018'] #utils.npy_block_names(test_output_files)
        df1d        = Inference.time_series_eval_csv(test_output_files, test_block_names, 16)
        df1d.to_csv(test_bc_df1_name)
        #df2d.to_csv(test_bc_df2_name)

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        if 'name' in param_group and param_group['name'] == 'noise_sigma':
            continue
        param_group['lr'] = lr

def cost_function(ytrue_w, ytrue, ypred_w, ypred):

    D_w = (ytrue_w[:, :, :] @ (ypred_w[:, :, :]).transpose(1, 2))[:, :, :, None]
    C_ij = ((ytrue[:, :,  None,:] - ypred[:, None,:,:]) ** 2).sum(-1) / 2
    C_ij = C_ij[:, :, :, None]  # reshape as a (N, M, 1) Tensor
    C_ij = C_ij*D_w

    return C_ij

def train(data_loader_training, data_loader_validate, model, optimizer, epochs, loss_stop_tolerance, criterion: str, best_model_name: str):

    best_val_loss = 100000000 # initial dummy value
    early_stopping = EarlyStopping(tolerance = loss_stop_tolerance, min_delta=50)
    

    loss_stats = {'train': [],"val": []}
    for epoch in range(1, epochs+1):

        training_start_time = time.time()
        train_epoch_loss = 0
        model.train()

        for batch, sample in enumerate(data_loader_training):
            
            Xtrain         = sample['image'].to(device)
            ytrain_true    = sample['mask'][:,:,:,:,0].to(device)
            EmbTrain       = sample['EmbMatrix'].to(device)
            WgTrain        = sample['weight'].to(device)
            #BWgt           = sample['batch_w'].to(device)

            list_ytrain_pred  = model(Xtrain) #model(Xtrain, EmbTrain)
            train_loss_w = 0
            optimizer.zero_grad()
            for l in range(15):
                if criterion      == 'mse': 
                    train_loss_   = F.mse_loss(ytrain_true, list_ytrain_pred[l])
                    train_loss_w  += train_loss_

                elif criterion == 'wmse':
                    train_loss_  = weighted_mse_loss(ytrain_true, list_ytrain_pred[l], WgTrain)
                    train_loss_w += train_loss_

                elif criterion == 'integral':
                    train_loss_ = weighted_integral_mse_loss(ytrain_true, list_ytrain_pred[l], WgTrain)
                    train_loss_w += train_loss_

                elif criterion == 'huber': 
                    train_loss_  = weighted_huber_mse_loss(ytrain_true, list_ytrain_pred[l], WgTrain)
                    train_loss_w += train_loss_

                elif criterion == 'wass':
                    #ytrain_true_ = ytrain_true[:, 0, :, :]
                    #ytrain_pred  = list_ytrain_pred[l][:, 0, :, :]
                    #train_weight = WgTrain[:, 0, :, :]
                    ytrain_true_      = torch.reshape(ytrain_true, (ytrain_true.shape[0], ytrain_true.shape[1]*ytrain_true.shape[2]*ytrain_true.shape[3], 1))
                    ytrain_pred       = torch.reshape(list_ytrain_pred[l], (list_ytrain_pred[l].shape[0], list_ytrain_pred[l].shape[1]*list_ytrain_pred[l].shape[2]*list_ytrain_pred[l].shape[3], 1))
                    #pred_train_weight = torch.reshape(WgTrain, (WgTrain.shape[0], WgTrain.shape[1]*WgTrain.shape[2]*WgTrain.shape[3], 1))


                    loss = SamplesLoss(loss="sinkhorn", p=2, blur = .05)

                    #train_loss_  = loss(pred_train_weight, ytrain_true_, pred_train_weight, ytrain_pred)
                    #train_loss_  = custom_wasserstien_loss(pred_train_weight, ytrain_true_, pred_train_weight, ytrain_pred, blur = 0.05, sinkhorn_nits = 20, weighted_cost_func = True)
                    train_loss_  = loss(ytrain_true_, ytrain_pred)
                    train_loss_  = torch.mean(train_loss_)
                    train_loss_w += train_loss_


            if criterion == 'integral':
                train_loss_w.requires_grad = True
            train_loss_w.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss_w.item() 

        # VALIDATION    
        with torch.no_grad():
            
            val_epoch_loss = 0
            
            model.eval()
            for batch, sample in enumerate(data_loader_validate):
                
                Xvalid      = sample['image'].to(device)
                yvalid_true = sample['mask'][:,:,:,:,0].to(device)
                EmbValid    = sample['EmbMatrix'].to(device)
                WgValid     = sample['weight'].to(device)
                #VBWgt       = sample['batch_w'].to(device)
        
                list_yvalid_pred   = model(Xvalid) #, EmbValid
                val_loss_sum_week = 0
                for l in range(len(list_yvalid_pred)):
                    if criterion == 'mse': 
                        val_loss_w = F.mse_loss(yvalid_true, list_yvalid_pred[l])
                        val_loss_sum_week += val_loss_w

                    elif criterion == 'wmse':
                        val_loss_w = weighted_mse_loss(yvalid_true, list_yvalid_pred[l], WgValid)
                        val_loss_sum_week += val_loss_w

                    elif criterion == 'integral':
                        val_loss_w = weighted_integral_mse_loss(yvalid_true, list_yvalid_pred[l], WgValid)
                        val_loss_sum_week += val_loss_w

                    elif criterion == 'huber': 
                        val_loss_w  = weighted_huber_mse_loss(yvalid_true, list_yvalid_pred[l], WgValid)
                        val_loss_sum_week += val_loss_w
                    elif criterion == 'wass':

                        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05) 
                        #yvalid_true_ = yvalid_true[:, 0, :, :]
                        #yvalid_pred  = list_yvalid_pred[l][:, 0, :, :]

                        yvalid_true_ = torch.reshape(yvalid_true, (yvalid_true.shape[0], yvalid_true.shape[1]*yvalid_true.shape[2]*yvalid_true.shape[3], 1))
                        yvalid_pred  = torch.reshape(list_yvalid_pred[l], (list_yvalid_pred[l].shape[0], list_yvalid_pred[l].shape[1]*list_yvalid_pred[l].shape[2]*list_yvalid_pred[l].shape[3], 1))
                        #valid_weight_pred = torch.reshape(WgValid, (WgValid.shape[0], WgValid.shape[1]*WgValid.shape[2]*WgValid.shape[3], 1))
                        #val_loss_w  = loss(valid_weight_pred, yvalid_true_, valid_weight_pred, yvalid_pred)
                        #val_loss_w  = custom_wasserstien_loss(valid_weight_pred, yvalid_true_, valid_weight_pred, yvalid_pred, blur = 0.05, sinkhorn_nits = 100, weighted_cost_func = True)
                        
                        val_loss_w  = loss(yvalid_true_,  yvalid_pred)
                        val_loss_w  = torch.mean(val_loss_w)
                        val_loss_sum_week += val_loss_w

                val_epoch_loss += val_loss_sum_week.item()

        loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
        loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))


        training_duration_time = (time.time() - training_start_time)        
        print(f'Epoch {epoch+0:03}: | Time(s): {training_duration_time:.3f}| Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 
        
        if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                    
            best_val_loss=(val_epoch_loss/len(data_loader_validate))
            torch.save(model.state_dict(), best_model_name)
            
            status = True

            print(f'Best model Saved! Val MSE: {best_val_loss:.4f}')
        else:
            print(f'Model is not saved! Current val Loss: {(val_epoch_loss/len(data_loader_validate)):.4f}') 
                
            status = False
            # early stopping
        early_stopping(status)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break

        
    return loss_stats



class ImbYieldEst():
    def __init__(self, model,  lr: float, wd: float, exp: str):

        self.model = model
        self.lr = lr
        self.wd = wd
        self.exp = exp

        params          = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer  = torch.optim.AdamW(params, lr = self.lr, betas=(0.9, 0.95), weight_decay = self.wd)


        self.exp_output_dir = '/data2/hkaman/DiT/EXPs/' + 'EXP_' + self.exp
        isExist  = os.path.isdir(self.exp_output_dir)

        if not isExist:
            os.makedirs(self.exp_output_dir)
            os.makedirs(self.exp_output_dir + '/coords')
            os.makedirs(self.exp_output_dir + '/loss')
            os.makedirs(self.exp_output_dir + '/checkpoints')

        self.best_model_name      = self.exp_output_dir + '/best_model_' + self.exp + '.pth'
        self.checkpoint_dir       = self.exp_output_dir + '/checkpoints/'
        self.loss_fig_name        = self.exp_output_dir + '/loss/loss_'  + self.exp + '.png'
        self.loss_df_name         = self.exp_output_dir + '/loss/loss_'  + self.exp + '.csv' 


            
        self.train_df_name        = self.exp_output_dir + '/' + self.exp + '_train.csv'
        self.valid_df_name        = self.exp_output_dir + '/' + self.exp + '_valid.csv'
        self.test_df_name         = self.exp_output_dir + '/' + self.exp + '_test.csv'
        self.timeseries_fig       = self.exp_output_dir + '/' + self.exp + '_timeseries.png'
        self.scatterplot          = self.exp_output_dir + '/' + self.exp + '_scatterplot.png'

    def train(self, data_loader_training, data_loader_validate, loss: str, epochs: int, loss_stop_tolerance: int):

        best_val_loss  = 100000000 # initial dummy value
        early_stopping = EarlyStopping(tolerance = loss_stop_tolerance, min_delta = 50)
        

        loss_stats = {'train': [],"val": []}


        for epoch in range(1, epochs+1):

            training_start_time = time.time()
            train_epoch_loss = 0

            self.model.train()

            for batch, sample in enumerate(data_loader_training):
                
                Xtrain         = sample['image'].to(device)
                ytrain_true    = sample['mask'][:,:,:,:,0].to(device)
                WgTrain        = sample['weight'].to(device)
                Emb_List       = sample["EmbList"]#.to(device)

                list_ytrain_pred  = self.model(Xtrain, Emb_List) #model(Xtrain, EmbTrain)

                self.optimizer.zero_grad()

                if loss   == 'mse': 
                    train_loss_   = mse_loss(ytrain_true, list_ytrain_pred)

                elif loss == 'wmse':
                    train_loss_  = weighted_mse_loss(ytrain_true, list_ytrain_pred, WgTrain)

                elif loss == 'huber': 
                    train_loss_  = weighted_huber_mse_loss(ytrain_true, list_ytrain_pred, WgTrain)

                elif loss == 'wass':

                    ytrain_true_      = torch.reshape(ytrain_true, (ytrain_true.shape[0], ytrain_true.shape[1]*ytrain_true.shape[2]*ytrain_true.shape[3], 1))
                    ytrain_pred       = torch.reshape(list_ytrain_pred, (list_ytrain_pred.shape[0], list_ytrain_pred.shape[1]*list_ytrain_pred.shape[2]*list_ytrain_pred.shape[3], 1))
 
                    loss = SamplesLoss(loss="sinkhorn", p=2, blur = .05)
                    train_loss_  = loss(ytrain_true_, ytrain_pred)
                    train_loss_  = torch.mean(train_loss_)


                train_loss_.backward()
                self.optimizer.step()
                # Anneal the learning rate (e.g., reduce by a factor)
                self.lr *= 0.95   
                # Warm-up the learning rate (e.g., increase linearly)
                # self.lr += 0.0001
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                
                train_epoch_loss += train_loss_.item() 

            # VALIDATION    
            with torch.no_grad():
                
                val_epoch_loss = 0
                
                #self.model.eval()
                for batch, sample in enumerate(data_loader_validate):
                    
                    Xvalid      = sample['image'].to(device)
                    yvalid_true = sample['mask'][:,:,:,:,0].to(device)
                    EmbList     = sample['EmbList']#.to(device)
                    WgValid     = sample['weight'].to(device)

                    list_yvalid_pred   = self.model(Xvalid, EmbList) 

                    if loss == 'mse': 
                        val_loss_w = mse_loss(yvalid_true, list_yvalid_pred)

                    elif loss == 'wmse':
                        val_loss_w = weighted_mse_loss(yvalid_true, list_yvalid_pred, WgValid)

                    elif loss == 'huber': 
                        val_loss_w  = weighted_huber_mse_loss(yvalid_true, list_yvalid_pred, WgValid)

                    elif loss == 'wass':

                        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05) 
                        yvalid_true_ = torch.reshape(yvalid_true, (yvalid_true.shape[0], yvalid_true.shape[1]*yvalid_true.shape[2]*yvalid_true.shape[3], 1))
                        yvalid_pred  = torch.reshape(list_yvalid_pred, (list_yvalid_pred.shape[0], list_yvalid_pred.shape[1]*list_yvalid_pred.shape[2]*list_yvalid_pred.shape[3], 1))
                        val_loss_w  = loss(yvalid_true_,  yvalid_pred)
                        val_loss_w  = torch.mean(val_loss_w)

                    val_epoch_loss += val_loss_w.item()

            loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
            loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))

            training_duration_time = (time.time() - training_start_time)        
            print(f'Epoch {epoch+0:03}: | Time(s): {training_duration_time:.3f}| Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 
            
            if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                        
                best_val_loss=(val_epoch_loss/len(data_loader_validate))
                torch.save(self.model.state_dict(), self.best_model_name)
                status = True
                print(f'Best model Saved! Val MSE: {best_val_loss:.4f}')
            else:
                print(f'Model is not saved! Current val Loss: {(val_epoch_loss/len(data_loader_validate)):.4f}') 
                status = False

            early_stopping(status)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
        _ = save_loss_df(loss_stats, self.loss_df_name, self.loss_fig_name)

    def predict(self, model, data_loader_training, data_loader_validate, data_loader_testing):
        # print(self.best_model_name)
        best_model_name = self.exp_output_dir + '/best_model_' + self.exp + '.pth'
        model.load_state_dict(torch.load(best_model_name))

        train_output_files = []
        valid_output_files = []
        test_output_files = []

        # with torch.no_grad():
            
            # self.model.eval()
            # # ================ Train===========================
        for batch, sample in enumerate(data_loader_training):
            
            img_train           = sample['image'].to(device)
            label_train_true    = sample['mask']
            EmbList             = sample['EmbList']
            ID_batch_train      = sample['block']
            Cult_batch_train    = sample['cultivar']
            Xcoord_batch_train  = sample['X']
            ycoord_batch_train  = sample['Y']
        
            
            label_train_pred = self.model(img_train, EmbList)
            
            label_train_true = label_train_true.detach().cpu().numpy()
            label_train_pred = label_train_pred.detach().cpu().numpy()


            this_batch_train = {"block": ID_batch_train, "cultivar": Cult_batch_train, "X": Xcoord_batch_train, "Y": ycoord_batch_train,
                                "ytrue": label_train_true, "ypred_w15": label_train_pred}
            
            train_output_files.append(this_batch_train)

        df_train = self._return_pred_df(train_output_files, None, 16)
        df_train.to_csv(self.train_df_name)
        print("Train evaluation is done!")

        # #================== Validaiton====================
        for batch, sample in enumerate(data_loader_validate):
            
            img_valid         = sample['image'].to(device)
            label_valid_true  = sample['mask']
            EmbList_val       = sample['EmbList']
            ID_batch_val      = sample['block']
            Cult_batch_val    = sample['cultivar']
            Xcoord_batch_val  = sample['X']
            ycoord_batch_val  = sample['Y']


            label_valid_pred = self.model(img_valid, EmbList_val)
                
            label_valid_true = label_valid_true.detach().cpu().numpy()
            label_valid_pred = label_valid_pred.detach().cpu().numpy()
            

            this_batch_val = {"block": ID_batch_val, "cultivar": Cult_batch_val, "X": Xcoord_batch_val, 
                              "Y": ycoord_batch_val, "ytrue": label_valid_true, "ypred_w15": label_valid_pred} 

                
            valid_output_files.append(this_batch_val)


        df_valid = self._return_pred_df(valid_output_files, None, 16)
        df_valid.to_csv(self.valid_df_name)

        print("validation evaluation is done!")
        # =================== Test ========================
        for batch, sample in enumerate(data_loader_testing):
            
            img_test           = sample['image'].to(device)
            label_test_true    = sample['mask'].to(device)
            EmbList_test       = sample['EmbList']
            ID_batch_test      = sample['block']
            Cult_batch_test    = sample['cultivar']
            Xcoord_batch_test  = sample['X']
            ycoord_batch_test  = sample['Y']


            label_test_pred = self.model(img_test, EmbList_test)
            
            label_test_true = label_test_true.detach().cpu().numpy()
            label_test_pred = label_test_pred.detach().cpu().numpy()


            this_batch_test = {"block": ID_batch_test, "cultivar": Cult_batch_test, "X": Xcoord_batch_test, "Y": ycoord_batch_test, 
                            "ytrue": label_test_true, "ypred_w15": label_test_pred}
            
            
            test_output_files.append(this_batch_test)

        #test_block_names  = ['LIV_186_2017', 'LIV_025_2019','LIV_105_2018', 'LIV_105_2019', 'LIV_181_2017', 'LIV_181_2018'] 
        df_test   = self._return_pred_df(test_output_files, None, 16)
        df_test.to_csv(self.test_df_name)
        print("Test evaluation is done!")

    def _return_pred_df(self, pred_npy, blocks_list, wsize = None):

        if blocks_list is None: 
            all_block_names = [dict['block'] for dict in pred_npy]#[0]

            blocks_list = list(set(item for sublist in all_block_names for item in sublist))
            # print(len(blocks_list))

        OutDF = pd.DataFrame()
        out_ytrue, out_blocks, out_cultivars, out_x, out_y = [], [], [], [], []
        out_ypred_w15 = []
        
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
                        x_vector, y_vector = self.xy_vector_generator(x0, y0, wsize)
                        out_x.append(x_vector)
                        out_y.append(y_vector)
        
                        tb_ytrue         = pred_npy[l]['ytrue'][index]
                        tb_flatten_ytrue = tb_ytrue.flatten()
                        out_ytrue.append(tb_flatten_ytrue)

                        tb_ypred_w15    = pred_npy[l]['ypred_w15'][index]
                        tb_flatten_ypred_w15 = tb_ypred_w15.flatten()
                        out_ypred_w15.append(tb_flatten_ypred_w15)
                        
                        tb_block_id   = np.array(len(tb_flatten_ytrue)*[block_id], dtype=np.int32)
                        out_blocks.append(tb_block_id)

                        tb_cultivar_id = np.array(len(tb_flatten_ytrue)*[cultivar_id], dtype=np.int8)
                        out_cultivars.append(tb_cultivar_id)

        out_blocks        = np.concatenate(out_blocks)
        out_cultivars     = np.concatenate(out_cultivars)
        out_x             = np.concatenate(out_x)
        out_y             = np.concatenate(out_y)
        out_ytrue         = np.concatenate(out_ytrue)
        out_ypred_w15     = np.concatenate(out_ypred_w15)
        
        OutDF['block']    = out_blocks
        OutDF['cultivar'] = out_cultivars
        OutDF['x']        = out_x
        OutDF['y']        = out_y
        OutDF['ytrue']    = out_ytrue
        OutDF['ypred_w15']    = out_ypred_w15
        
        return OutDF

    def xy_vector_generator(self, x0, y0, wsize):
        x_vector, y_vector = [], []
        
        for i in range(x0, x0+wsize):
            for j in range(y0, y0+wsize):
                x_vector.append(i)
                y_vector.append(j)

        return x_vector, y_vector 