import os
import numpy as np 
import pandas as pd
import time
import torch
import torch.nn as nn
from src.models import UNet2DConvLSTM

from src import utils, ModelEngine, dataloader
from src.RWSampler import lds_prepare_weights, return_cost_sensitive_weight_sampler
from src.losses import *


device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
#==============================================================================================================#
#==================================================Initialization =============================================#
#==============================================================================================================#
def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        if 'name' in param_group and param_group['name'] == 'noise_sigma':
            continue
        param_group['lr'] = lr



def train(data_loader_training, data_loader_validate, model, optimizer, epochs, loss_stop_tolerance, lr, criterion, best_model_name):
    best_val_loss = 100000 # initial dummy value
    early_stopping = ModelEngine.EarlyStopping(tolerance = loss_stop_tolerance, min_delta=50)
    
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
                
            list_ytrain_pred  = model(Xtrain, EmbTrain)
            train_loss_w = 0
            optimizer.zero_grad()
            for l in range(15):
                #val_loss = criterion(ytrain_true, list_ytrain_pred[l])
                train_loss_ = weighted_huber_mse_loss(ytrain_true, list_ytrain_pred[l], WgTrain)
                train_loss_w += train_loss_

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
        
                list_yvalid_pred   = model(Xvalid, EmbValid)
                val_loss_sum_week = 0
                for l in range(len(list_yvalid_pred)):
                    #val_loss = criterion(yvalid_true, list_yvalid_pred[l])
                    val_loss_w = weighted_huber_mse_loss(yvalid_true, list_yvalid_pred[l], WgValid)
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


def run(batch_size: int, dropout: int, 
            learning_rate: float, weight_decay: float,
            epochs: int, loss_stop_tolerance: int, 
            lds_ks: int, lds_sigma: int, dw_alpha :float, betha: int, init_noise_sigma: float, sigma_lr: float,
            re_weighting_method: str,
            exp_name: str):

    exp_output_dir = '/data2/hkaman/Imbalance/EXPs/' + 'EXP_' + exp_name
    isExist  = os.path.isdir(exp_output_dir)

    if not isExist:
        os.makedirs(exp_output_dir)
        os.makedirs(exp_output_dir + '/coords')
        os.makedirs(exp_output_dir + '/loss')

    best_model_name      = exp_output_dir + '/best_model' + exp_name + '.pth'
    loss_fig_name        = exp_output_dir + '/loss/loss'  + exp_name + '.png'
    loss_df_name         = exp_output_dir + '/loss/loss'  + exp_name + '.csv' 


    data_loader_training, data_loader_validate, data_loader_test  = dataloader.getData(batch_size, 
                                                                                                  lds_ks, 
                                                                                                  lds_sigma, 
                                                                                                  dw_alpha, 
                                                                                                  betha, 
                                                                                                  re_weighting_method = re_weighting_method, 
                                                                                                  exp_name= exp_name)


    model = UNet2DConvLSTM(in_channels = 6, out_channels = 1, 
                                num_filters   = 16, 
                                dropout       = dropout, 
                                Emb_Channels  = 4, 
                                batch_size    = batch_size, 
                                botneck_size  = 2).to(device)


    #criterion = nn.MSELoss()
    criterion = BMCLoss(init_noise_sigma)
    

    ### ADAM
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay = weight_decay)
    optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': sigma_lr, 'name': 'noise_sigma'})
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
   

    loss_stats = train(data_loader_training, data_loader_validate, model, optimizer, epochs, loss_stop_tolerance, learning_rate, criterion, best_model_name)

    _ = ModelEngine.save_loss_df(loss_stats, loss_df_name, loss_fig_name)
    _ = ModelEngine.predict(model, data_loader_training, data_loader_validate, data_loader_test, Exp_name = exp_name)


if __name__ == "__main__":
    #lds_kds = [5, 10, 15, 20]
    #lds_sigmas = [2, 4, 6, 8]
    alpha_list = [3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]

    for a in alpha_list: 
        ExpName = '015_64_001_05_RGB_DW_M_' + str(a)
        run(batch_size = 64, dropout = 0.3, 
            learning_rate = 0.001, weight_decay = 0.05,
            epochs = 500, loss_stop_tolerance = 100, 
            lds_ks = 10, lds_sigma = 8, dw_alpha = a, betha = 4, init_noise_sigma = 1.0, sigma_lr = 1e-2,
            re_weighting_method = 'dw',
            exp_name = ExpName) 
