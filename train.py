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


def run(batch_size: int, dropout: int, 
            learning_rate: float, weight_decay: float,
            epochs: int, loss_stop_tolerance: int, 
            lds_ks: int, lds_sigma: int, dw_alpha :float, betha: int, init_noise_sigma: float, sigma_lr: float,
            re_weighting_method: str, criterion: str,
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


    params     = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(params, lr=learning_rate, weight_decay = weight_decay)
    #loss_stats = ModelEngine.train(data_loader_training, data_loader_validate, model, optimizer, epochs, loss_stop_tolerance, criterion = criterion, best_model_name = best_model_name)
    #_          = ModelEngine.save_loss_df(loss_stats, loss_df_name, loss_fig_name)
    _          = ModelEngine.predict(model, data_loader_training, data_loader_validate, data_loader_test, Exp_name = exp_name)


if __name__ == "__main__":
    #lds_kds = [5, 10, 15, 20]
    #lds_sigmas = [2, 4, 6, 8]
    #alpha_list = [3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
    #for a in alpha_list: 
    ExpName = '016_64_001_05_WD_Resample_' + str(3.9) + '_huberLoss'
    run(batch_size = 64, dropout = 0.3, 
        learning_rate = 0.001, weight_decay = 0.05,
        epochs = 500, loss_stop_tolerance = 100, 
        lds_ks = 10, lds_sigma = 8, dw_alpha = 3.9, betha = 4, init_noise_sigma = 1.0, sigma_lr = 1e-2,
        re_weighting_method = 'dw', criterion = 'huber', 
        exp_name = ExpName) 
