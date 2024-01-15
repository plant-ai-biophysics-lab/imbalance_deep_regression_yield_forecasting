import torch
import argparse
import numpy as np
# Import custom dataloader, model, and engine modules
from utils import dataloader
from models import configs, engine


from models.CNNs import UNet2DConvLSTM

def main(args):
    # fix the seed for reproducibility
    seed = 1987 + engine.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Check if there is GPU(s): {torch.cuda.is_available()}")


    Exp_name = args.exp_name
    batch_size = args.batch_size
    in_channels = args.in_channels
    dropout = args.dropout
    lds_ks = args.ldsks
    lds_sigma = args.ldssigma
    dw_alpha = args.alphs
    cb_betha = args.betha
    lr = args.lr
    wd = args.wd
    loss = args.loss
    epochs = args.epochs
    reweight = args.reweight
    resampling = args.resampling
    cond_status = args.cond


    data_loader_training, data_loader_validate, data_loader_test = dataloader.dataloaders(
        batch_size = batch_size, 
        in_channels = in_channels, 
        lds_ks = lds_ks,
        lds_sigma = lds_sigma, 
        dw_alpha = dw_alpha, 
        betha = cb_betha, 
        reweighting_method = reweight, 
        resmapling_status = resampling,
        exp_name = Exp_name
        )

    # Create model configuration using custom configs module
    config = configs.build_configs(
        img_size = 16, 
        in_channels = in_channels,
        out_channels = 1, 
        cond = cond_status,
        dropout = dropout, 
        ).call()
    
    # Create and move model to GPU
    model = UNet2DConvLSTM(config, cond = cond_status).to(device)
    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    # Create an instance of ImbYieldEst engine
    YE = engine.YieldEst(model, 
                                 lr = lr, 
                                 wd = wd, 
                                 exp = Exp_name)
    # Train the model
    _ = YE.train(data_loader_training, data_loader_validate, 
                      loss = loss, 
                      epochs = epochs, 
                      loss_stop_tolerance = 100)
    # Predict 
    model_pred = UNet2DConvLSTM(config, cond = False).to(device)
    # _ = YE.predict(model, data_loader_training, data_loader_validate, data_loader_test)
    _ = YE.predict(model_pred, data_loader_training, category= 'train', iter = 1)
    _ = YE.predict(model_pred, data_loader_validate, category= 'valid', iter = 1)
    _ = YE.predict(model_pred, data_loader_test, category= 'test', iter = 10)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Imbalance Deep Yield Estimation")
    parser.add_argument("--exp_name",    type=str,   default = "test",help = "Experiment name")
    parser.add_argument("--batch_size",  type=int,   default = 64,   help = "Batch size")
    parser.add_argument("--in_channels", type=int,   default = 6,     help = "Number of input channels")
    parser.add_argument("--dropout",     type=float, default = 0.3,   help = "Amount of dropout")
    parser.add_argument("--ldsks",       type=int,   default = 10,    help = "value of kernel density of lds algorithm")
    parser.add_argument("--ldssigma",    type=int,   default = 8,     help = "Value of sigma for lds algorithm")
    parser.add_argument("--alphs",       type=float, default = 3.9,   help = "Value of alpha for dense weight algorithm")
    parser.add_argument("--betha",       type=int,   default = 3,     help = "Value of Betha for BC algorithm")
    parser.add_argument("--lr",          type=float, default = 0.001, help = "Learning rate")
    parser.add_argument("--wd",          type=float, default = 0.05,  help = "Value of weight decay")
    parser.add_argument("--epochs",      type=int,   default = 500,   help = "The number of epochs")
    parser.add_argument("--loss",        type=str,   default = "mse", help = "Loss function  mse wmse huber wass")
    parser.add_argument("--reweight",    type=str,   default = "dw",  help = "Reweight strategy") # "dw", "lds", "cb", ""
    parser.add_argument("--resampling",  type=str,   default = False, help = "Weight resampling status") 
    parser.add_argument("--cond",        type=str,   default = False, help = "Conditional Model to use") 

    args = parser.parse_args()

    main(args)


