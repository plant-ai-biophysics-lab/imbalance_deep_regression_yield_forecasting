"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
import time
import argparse
import logging
import os

from latent_diffusion_tr.model.models import DiT
# from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusion_config import gaussian_diffusion as gd
import latent_diffusion_tr.utils.dl as dl

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_loss_df(loss_stat, loss_df_name, loss_fig_name):

    df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    df.to_csv(loss_df_name) 

    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    #plt.ylim(0, df['value'].max())
    plt.yscale('log')
    plt.savefig(loss_fig_name, dpi = 300)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
torch.manual_seed(1987)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params   = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag




# def main():
exp_name = 'Ex1_RGBN'
exp_output_dir = '/data2/hkaman/DiT/EXPs/' + 'EXP_' + exp_name
isExist  = os.path.isdir(exp_output_dir)

if not isExist:
    os.makedirs(exp_output_dir)
    os.makedirs(exp_output_dir + '/coords')
    os.makedirs(exp_output_dir + '/loss')
    os.makedirs(exp_output_dir + '/checkpoints')

best_model_name      = exp_output_dir + '/best_model_' + exp_name + '.pth'
loss_fig_name        = exp_output_dir + '/loss/loss_'  + exp_name + '.png'
loss_df_name         = exp_output_dir + '/loss/loss_'  + exp_name + '.csv' 

data_loader_training, data_loader_validate, data_loader_test  = dl.get_dataset(batch_size = 256, 
                                                                                        lds_ks = 10, 
                                                                                        lds_sigma = 8, 
                                                                                        dw_alpha = 3.9, 
                                                                                        betha = 4, 
                                                                                        re_weighting_method= 'dw', 
                                                                                        exp_name = exp_name)


model = DiT(input_size   = 16,
                    patch_size  = 4,
                    in_channels = 1,
                    hidden_size = 1024,
                    depth       = 28,
                    num_heads   = 16,
                    mlp_ratio   = 4.0,
                    learn_sigma = True,).to(device)

ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
requires_grad(ema, False)
vae = AutoencoderKL(in_channels = 1, out_channels = 1, latent_channels = 1).to(device) #.from_pretrained(f"stabilityai/sd-vae-ft-ema")


opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

# Prepare models for training:
update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
model.train()  # important! This enables embedding dropout for classifier-free guidance
ema.eval()  # EMA model should always be in eval mode

betas = gd.get_named_beta_schedule("linear", 1000)
loss_type = gd.LossType.MSE
timestep_respacing     = [1000]
use_kl                 = False
sigma_small            = False
predict_xstart         = False
learn_sigma            = True
rescale_learned_sigmas = False

log_every        = 100
ckpt_every       = 50

model_mean_type  = (gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X)
model_var_type   = ((gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL) if not learn_sigma else gd.ModelVarType.LEARNED_RANGE)

GaussianModel    = gd.GaussianDiffusion(betas, model_mean_type, model_var_type, loss_type)
loss_stats = {'train': []}

best_loss = 100000000

for epoch in range(300):
    training_start_time = time.time()
    running_loss = 0
    train_steps = 0
    log_steps = 0   

    for sample in data_loader_training:
        x_start  = sample['mask'].to(device) 
        cond     = sample['image'].to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x_start = vae.encode(x_start).latent_dist.sample().mul_(0.18215)

        t = torch.randint(0, 1000, (x_start.shape[0],), device=device)
        model_kwargs = dict(y=cond)


        loss_dict = GaussianModel.training_losses(model, x_start, t, model_kwargs = cond, noise = None)

        loss = loss_dict["loss"].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        #update_ema(ema, model.module)
        running_loss += loss.item()
        log_steps    += 1
        train_steps  += 1


    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
    }
    checkpoint_path = f"{exp_output_dir + '/checkpoints'}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    



    avg_loss = torch.tensor(running_loss / log_steps)
    loss_stats['train'].append(np.float32(avg_loss))

    training_duration_time = (time.time() - training_start_time)
    print(f'Epoch {epoch+0:03}: | Time(s): {training_duration_time:.3f}| Train Loss: {avg_loss:.4f}') 


    if (avg_loss) < best_loss or epoch==0:
                    
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_name)
            print(f'============================ Best Model is Saved! Train Loss: {avg_loss:.4f}')

_  = save_loss_df(loss_stats, loss_df_name, loss_fig_name)