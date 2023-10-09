import torch
import argparse

# Import custom dataloader, model, and engine modules
from utils import dataloader
from models.ViT import ours
from models.UNetConvLSTM import UNet2DConvLSTM
from models import configs, engine


def main(args):

    # seed = 1987
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)

    # Check if GPU is available, set device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Check if there is GPU(s): {torch.cuda.is_available()}")

    # Extract hyperparameters from command line arguments
    Exp_name    = args.exp_name
    batch_size  = args.batch_size
    embd_size   = args.embd_size
    num_heads   = args.num_heads
    num_layers  = args.num_layers
    in_channels = args.in_channels
    dropout     = args.dropout
    lds_ks      = args.ldsks
    lds_sigma   = args.ldssigma
    dw_alpha    = args.alphs
    cb_betha    = args.betha
    post_norm   = args.postnorm
    lr          = args.lr
    wd          = args.wd
    loss        = args.loss
    epochs      = args.epochs
    reweight    = args.reweight

    # Get data loaders from custom dataloader module
    data_loader_training, data_loader_validate, data_loader_test = dataloader.return_dataloaders(batch_size = batch_size, 
                                                                                                in_channels = in_channels, 
                                                                                                lds_ks      = lds_ks,
                                                                                                lds_sigma   = lds_sigma, 
                                                                                                dw_alpha    = dw_alpha, 
                                                                                                betha       = cb_betha, 
                                                                                                reweighting_method = reweight, 
                                                                                                resmapling_status = False,
                                                                                                exp_name = Exp_name)

    # Create model configuration using custom configs module
    config = configs.SRTR_Configs(img_size = 16, patch_size = 8, embed_dim = embd_size, mlp_dim = 512, 
                        in_channels = in_channels, out_channels = 1, num_heads = num_heads, num_layers = num_layers, cond = None,
                        Attn_drop = dropout, Proj_drop = dropout, PostNorm = post_norm, vis = True, 
                        kernel_size = [3, 3], dilation = [1, 1], stride  = [1, 1]
                        ).call()
    
    # Create and move model to the selected device (CPU or GPU)
    model = ours(config).to(device)

    # Create an instance of ImbYieldEst engine
    ImYiEst = engine.ImbYieldEst(model, 
                                 lr = lr, 
                                 wd = wd, 
                                 exp = Exp_name)
    # Train the model
    _ = ImYiEst.train(data_loader_training, data_loader_validate, 
                      loss = loss, 
                      epochs = epochs, 
                      loss_stop_tolerance = 50)
    # Predict 
    _ = ImYiEst.predict(model, data_loader_training, data_loader_validate, data_loader_test)

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Imbalance Deep Yield Estimation")
    parser.add_argument("--exp_name",    type=str,   default = "test",help = "Experiment name")
    parser.add_argument("--batch_size",  type=int,   default = 512,   help = "Batch size")
    parser.add_argument("--embd_size",   type=int,   default = 1024,  help = "Embedding size")
    parser.add_argument("--num_heads",   type=int,   default = 8,     help = "Number of attention heads")
    parser.add_argument("--num_layers",  type=int,   default = 6,     help = "Number of transformer layers")
    parser.add_argument("--in_channels", type=int,   default = 4,     help = "Number of input channels")
    parser.add_argument("--dropout",     type=float, default = 0.1,   help = "Amount of dropout")
    parser.add_argument("--ldsks",       type=int,   default = 10,    help = "value of kernel density of lds algorithm")
    parser.add_argument("--ldssigma",    type=int,   default = 8,     help = "Value of sigma for lds algorithm")
    parser.add_argument("--alphs",       type=float, default = 3.9,   help = "Value of alpha for dense weight algorithm")
    parser.add_argument("--betha",       type=int,   default = 4,     help = "Value of Betha for BC algorithm")
    parser.add_argument("--postnorm",    type=str,   default = False, help = "Post or Before Normalization for Self-Attention")
    parser.add_argument("--lr",          type=float, default = 0.0001, help = "Learning rate")
    parser.add_argument("--wd",          type=float, default = 0.0001,  help = "Value of weight decay")
    parser.add_argument("--epochs",      type=int,   default = 100,    help = "The number of epochs")
    parser.add_argument("--loss",        type=str,   default = "mse", help = "Loss function  mse wmse huber wass")
    parser.add_argument("--reweight",    type=str,   default = "dw",  help = "Reqeight strategy") # "dw", "lds", "cb"

    args = parser.parse_args()


    main(args)


