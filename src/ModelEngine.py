import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
from src import Inference 
from src.RWSampler import lds_prepare_weights, return_cost_sensitive_weight_sampler
from src.losses import *
from geomloss import SamplesLoss   # ImagesLoss
#======================================================================================================================================#
#=========================================================== 2D Config =================================================================
#======================================================================================================================================#

def Encoder2D(in_channels, middle_channels, out_channels, dropout):
    return nn.Sequential(
        nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(middle_channels),
        nn.PReLU(),
        nn.Dropout2d(p=dropout),
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(),
        nn.Dropout2d(p=dropout),)
# Decoder Block: 
def Decoder2D(in_channels, out_channels, dropout):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(),
        nn.Dropout2d(p=dropout),)

def MaxPolling2D():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def OutConv2D_3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.Upsample(scale_factor= 4, mode='bicubic'))

def OutConv2D_2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.Upsample(scale_factor= 2, mode='bicubic'))

def OutConv2D(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))       

class HadamardProduct(nn.Module):
    def __init__(self, shape):
        super(HadamardProduct, self).__init__()
        self.weights = nn.Parameter(torch.rand(shape)).cuda()
        
    def forward(self, x):
        return x*self.weights

class ConvLSTMCell(nn.Module):

    def __init__(self, img_size, img_width, input_dim, hidden_dim, kernel_size, 
                 cnn_dropout, rnn_dropout, bias=True, peephole=False,
                 layer_norm=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel for both cnn and rnn.
        cnn_dropout, rnn_dropout: float
            cnn_dropout: dropout rate for convolutional input.
            rnn_dropout: dropout rate for convolutional state.
        bias: bool
            Whether or not to add the bias.
        peephole: bool
            add connection between cell state to gates
        layer_norm: bool
            layer normalization 
        """

        super(ConvLSTMCell, self).__init__()
        self.input_shape = img_size
        self.img_width = img_width
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (int(self.kernel_size[0]/2), int(self.kernel_size[1]/2))
        self.stride = (1, 1)
        self.bias = bias
        self.peephole = peephole
        self.layer_norm = layer_norm
        
        self.out_height = int(self.img_width)
        self.out_width = int(self.img_width)
        
        self.input_conv = nn.Conv2d(in_channels=self.input_dim, out_channels=4*self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  stride = self.stride,
                                  padding=self.padding,
                                  bias=self.bias)
        self.rnn_conv = nn.Conv2d(self.hidden_dim, out_channels=4*self.hidden_dim, 
                                  kernel_size = self.kernel_size,
                                  padding=(math.floor(self.kernel_size[0]/2), 
                                         math.floor(self.kernel_size[1]/2)),
                                  bias=self.bias)
        
        if self.peephole is True:
            self.weight_ci = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.weight_cf = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.weight_co = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.layer_norm_ci = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
            self.layer_norm_cf = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
            self.layer_norm_co = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
        
            
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        
        self.layer_norm_x = nn.LayerNorm([4*self.hidden_dim, self.out_height, self.out_width])
        self.layer_norm_h = nn.LayerNorm([4*self.hidden_dim, self.out_height, self.out_width])
        self.layer_norm_cnext = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        x = self.cnn_dropout(input_tensor)
        x_conv = self.input_conv(x)
        if self.layer_norm is True:
            x_conv = self.layer_norm_x(x_conv)
        # separate i, f, c o
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_dim, dim=1)
        #print(f"{x_i.shape}|{x_f.shape}|{x_c.shape}|{x_o.shape}")
        
        h = self.rnn_dropout(h_cur)
        h_conv = self.rnn_conv(h)
        if self.layer_norm is True:
            h_conv = self.layer_norm_h(h_conv)
        # separate i, f, c o
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_dim, dim=1)
        #print(f"{h_i.shape}|{h_f.shape}|{h_c.shape}|{h_o.shape}")
    
        
        if self.peephole is True:
            f = torch.sigmoid((x_f + h_f) +  self.layer_norm_cf(self.weight_cf(c_cur)) if self.layer_norm is True else self.weight_cf(c_cur))
            i = torch.sigmoid((x_i + h_i) +  self.layer_norm_ci(self.weight_ci(c_cur)) if self.layer_norm is True else self.weight_ci(c_cur))
        else:
            
            f = torch.sigmoid((x_f + h_f))
            i = torch.sigmoid((x_i + h_i))
        
        
        g = torch.tanh((x_c + h_c))
        c_next = f * c_cur + i * g
        if self.peephole is True:
            o = torch.sigmoid(x_o + h_o + self.layer_norm_co(self.weight_co(c_cur)) if self.layer_norm is True else self.weight_co(c_cur))
        else:
            o = torch.sigmoid((x_o + h_o))
        
        if self.layer_norm is True:
            c_next = self.layer_norm_cnext(c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        height, width = self.out_height, self.out_width
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device))

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        cnn_dropout, rnn_dropout: float
            cnn_dropout: dropout rate for convolutional input.
            rnn_dropout: dropout rate for convolutional state.
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_sequence: return output sequence or final output only
        bidirectional: bool
            bidirectional ConvLSTM
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two sequences output and state
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(input_dim=64, hidden_dim=16, kernel_size=(3, 3), 
                               cnn_dropout = 0.2,
                               rnn_dropout=0.2, batch_first=True, bias=False)
        >> output, last_state = convlstm(x)
    """

    def __init__(self, img_size, img_width, input_dim, hidden_dim, kernel_size,
                 cnn_dropout=0.5, rnn_dropout=0.5,  
                 batch_first=False, bias=True, peephole=False,
                 layer_norm=False,
                 return_sequence=True,
                 bidirectional=False):
        super(ConvLSTM, self).__init__()

        #print(kernel_size)
        self.batch_first = batch_first
        self.return_sequence = return_sequence
        self.bidirectional = bidirectional

        cell_fw = ConvLSTMCell(img_size = img_size,
                               img_width = img_width, 
                                 input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 cnn_dropout=cnn_dropout,
                                 rnn_dropout=rnn_dropout,
                                 bias=bias,
                                 peephole=peephole,
                                 layer_norm=layer_norm)
        self.cell_fw = cell_fw
        
        if self.bidirectional is True:
            cell_bw = ConvLSTMCell(img_size = img_size,
                                   img_width = img_width,
                                     input_dim=input_dim,
                                     hidden_dim=hidden_dim,
                                     kernel_size=kernel_size,
                                     cnn_dropout=cnn_dropout,
                                     rnn_dropout=rnn_dropout,
                                     bias=bias,
                                     peephole=peephole,
                                     layer_norm=layer_norm)
            self.cell_bw = cell_bw

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        layer_output, last_state
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()
        #print(f"{b}|{seq_len}")
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state, hidden_state_inv = self._init_hidden(batch_size=b)
            # if self.bidirectional is True:
            #     hidden_state_inv = self._init_hidden(batch_size=b)

        ## LSTM forward direction
        input_fw = input_tensor
        
        h, c = hidden_state
        #print(f"I am here: {h.shape}|{c.shape}")
        
        
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell_fw(input_tensor = input_fw[:, t, :, :, :],
                                             cur_state=[h, c])
            
            output_inner.append(h)
        output_inner = torch.stack((output_inner), dim=1)
        layer_output = output_inner
        last_state = [h, c]
        ####################
        
        
        ## LSTM inverse direction
        if self.bidirectional is True:
            input_inv = input_tensor
            h_inv, c_inv = hidden_state_inv
            output_inv = []
            for t in range(seq_len-1, -1, -1):
                h_inv, c_inv = self.cell_bw(input_tensor=input_inv[:, t, :, :, :],
                                                 cur_state=[h_inv, c_inv])
                
                output_inv.append(h_inv)
            output_inv.reverse() 
            output_inv = torch.stack((output_inv), dim=1)
            layer_output = torch.cat((output_inner, output_inv), dim=2)
            last_state_inv = [h_inv, c_inv]
        ###################################
        
        #return layer_output if self.return_sequence is True else layer_output[:, -1:], last_state, last_state_inv if self.bidirectional is True else None
        return layer_output, last_state

    def _init_hidden(self, batch_size):
        init_states_fw = self.cell_fw.init_hidden(batch_size)
        init_states_bw = None
        if self.bidirectional is True:
            init_states_bw = self.cell_bw.init_hidden(batch_size)
        return init_states_fw, init_states_bw  

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
        for batch, sample in enumerate(data_loader_training):
            
            X_batch_train       = sample['image'].to(device)
            y_batch_train       = sample['mask'].to(device)
            C_batch_train       = sample['EmbMatrix'].to(device)
            ID_batch_train      = sample['block']
            Cult_batch_train    = sample['cultivar']
            Xcoord_batch_train  = sample['X']
            ycoord_batch_train  = sample['Y']
        
            
            list_y_train_pred = model(X_batch_train, C_batch_train)
            
            y_true_train = y_batch_train.detach().cpu().numpy()
            
            ytpw1 = list_y_train_pred[0].detach().cpu().numpy()
            ytpw2 = list_y_train_pred[1].detach().cpu().numpy()
            ytpw3 = list_y_train_pred[2].detach().cpu().numpy()
            ytpw4 = list_y_train_pred[3].detach().cpu().numpy()
            ytpw5 = list_y_train_pred[4].detach().cpu().numpy()
            ytpw6 = list_y_train_pred[5].detach().cpu().numpy()
            ytpw7 = list_y_train_pred[6].detach().cpu().numpy()
            ytpw8 = list_y_train_pred[7].detach().cpu().numpy()
            ytpw9 = list_y_train_pred[8].detach().cpu().numpy()
            ytpw10 = list_y_train_pred[9].detach().cpu().numpy()
            ytpw11 = list_y_train_pred[10].detach().cpu().numpy()
            ytpw12 = list_y_train_pred[11].detach().cpu().numpy()
            ytpw13 = list_y_train_pred[12].detach().cpu().numpy()
            ytpw14 = list_y_train_pred[13].detach().cpu().numpy()
            ytpw15 = list_y_train_pred[14].detach().cpu().numpy()


            this_batch_train = {"block": ID_batch_train, "cultivar": Cult_batch_train, "X": Xcoord_batch_train, "Y": ycoord_batch_train,
                                "ytrue": y_true_train, "ypred_w1": ytpw1, "ypred_w2": ytpw2,"ypred_w3": ytpw3,"ypred_w4": ytpw4,"ypred_w5": ytpw5,"ypred_w6": ytpw6,"ypred_w7": ytpw7,"ypred_w8": ytpw8,
                                "ypred_w9": ytpw9,"ypred_w10": ytpw10,"ypred_w11": ytpw11,"ypred_w12": ytpw12,"ypred_w13": ytpw13,"ypred_w14": ytpw14,"ypred_w15": ytpw15}
            
            train_output_files.append(this_batch_train)

        train_df = Inference.ScenarioEvaluation2D(train_output_files)
        train_df.to_csv(train_df_name)
        print("train evaluation is done!")
        #train_block_names      = utils.npy_block_names(train_output_files)
        #df1d_train, df2d_train = utils.time_series_eval_csv(train_output_files, train_block_names, patch_size)
        #df1d_train.to_csv(train_bc_df1_name)
        #df2d_train.to_csv(train_bc_df2_name)

        #================== Validaiton====================
        for batch, sample in enumerate(data_loader_validate):
            
            X_batch_val       = sample['image'].to(device)
            y_batch_val       = sample['mask'].to(device)
            C_batch_val       = sample['EmbMatrix'].to(device)
            ID_batch_val      = sample['block']
            Cult_batch_val    = sample['cultivar']
            Xcoord_batch_val  = sample['X']
            ycoord_batch_val  = sample['Y']


            list_y_val_pred = model(X_batch_val, C_batch_val)
                
            y_true_val    = y_batch_val.detach().cpu().numpy()

            yvpw1 = list_y_val_pred[0].detach().cpu().numpy()
            yvpw2 = list_y_val_pred[1].detach().cpu().numpy()
            yvpw3 = list_y_val_pred[2].detach().cpu().numpy()
            yvpw4 = list_y_val_pred[3].detach().cpu().numpy()
            yvpw5 = list_y_val_pred[4].detach().cpu().numpy()
            yvpw6 = list_y_val_pred[5].detach().cpu().numpy()
            yvpw7 = list_y_val_pred[6].detach().cpu().numpy()
            yvpw8 = list_y_val_pred[7].detach().cpu().numpy()
            yvpw9 = list_y_val_pred[8].detach().cpu().numpy()
            yvpw10 = list_y_val_pred[9].detach().cpu().numpy()
            yvpw11 = list_y_val_pred[10].detach().cpu().numpy()
            yvpw12 = list_y_val_pred[11].detach().cpu().numpy()
            yvpw13 = list_y_val_pred[12].detach().cpu().numpy()
            yvpw14 = list_y_val_pred[13].detach().cpu().numpy()
            yvpw15 = list_y_val_pred[14].detach().cpu().numpy()
            

            this_batch_val = {"block": ID_batch_val, "cultivar": Cult_batch_val, "X": Xcoord_batch_val, "Y": ycoord_batch_val, "ytrue": y_true_val, "ypred_w1": yvpw1, "ypred_w2": yvpw2, "ypred_w3": yvpw3, "ypred_w4": yvpw4, "ypred_w5": yvpw5, "ypred_w6": yvpw6, "ypred_w7": yvpw7, "ypred_w8": yvpw8,
                                "ypred_w9": yvpw9, "ypred_w10": yvpw10, "ypred_w11": yvpw11, "ypred_w12": yvpw12, "ypred_w13": yvpw13, "ypred_w14": yvpw14, "ypred_w15": yvpw15} 

                
            valid_output_files.append(this_batch_val)
        # save the prediction in data2 drectory as a npy file
        #np.save(valid_npy_name, valid_output_files)
        valid_df = Inference.ScenarioEvaluation2D(valid_output_files)
        valid_df.to_csv(valid_df_name)

        #valid_block_names  = utils.npy_block_names(valid_output_files)
        #df1d_valid, df2d_valid = utils.time_series_eval_csv(valid_output_files, valid_block_names, patch_size)
        #df1d_valid.to_csv(valid_bc_df1_name)
        #df2d_valid.to_csv(valid_bc_df2_name)
        print("validation evaluation is done!")
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
        test_block_names  = ['LIV_186_2017', 'LIV_025_2019', 'LIV_105_2018'] #utils.npy_block_names(test_output_files)
        df1d        = Inference.time_series_eval_csv(test_output_files, test_block_names, 16)
        df1d.to_csv(test_bc_df1_name)
        #df2d.to_csv(test_bc_df2_name)

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        if 'name' in param_group and param_group['name'] == 'noise_sigma':
            continue
        param_group['lr'] = lr


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
            print()
            list_ytrain_pred  = model(Xtrain, EmbTrain)
            train_loss_w = 0
            optimizer.zero_grad()
            for l in range(15):
                if criterion == 'mse': 
                    train_loss_ = F.mse_loss(ytrain_true, list_ytrain_pred[l])
                    train_loss_w += train_loss_

                elif criterion == 'wmse':
                    train_loss_ = weighted_mse_loss(ytrain_true, list_ytrain_pred[l], WgTrain)
                    train_loss_w += train_loss_

                elif criterion == 'integral':

                    train_loss_ = weighted_integral_mse_loss(ytrain_true, list_ytrain_pred[l], WgTrain)
                    train_loss_w += train_loss_

                elif criterion == 'huber': 
                    train_loss_  = weighted_huber_mse_loss(ytrain_true, list_ytrain_pred[l], WgTrain)
                    train_loss_w += train_loss_

                elif criterion == 'wass':
                    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05) 
                    #ytrain_true_ = ytrain_true[:, 0, :, :]
                    #ytrain_pred  = list_ytrain_pred[l][:, 0, :, :]
                    #train_weight = WgTrain[:, 0, :, :]
                    ytrain_true_ = torch.reshape(ytrain_true, (ytrain_true.shape[0], ytrain_true.shape[1]*ytrain_true.shape[2]*ytrain_true.shape[3], 1))
                    ytrain_pred  = torch.reshape(list_ytrain_pred[l], (list_ytrain_pred[l].shape[0], list_ytrain_pred[l].shape[1]*list_ytrain_pred[l].shape[2]*list_ytrain_pred[l].shape[3], 1))
                    train_weight = torch.reshape(WgTrain, (WgTrain.shape[0], WgTrain.shape[1]*WgTrain.shape[2]*WgTrain.shape[3], 1))
                    #print(f"{ytrain_true_.shape} | {ytrain_pred.shape}")
                    train_loss_  = loss(train_weight, ytrain_true_, train_weight, ytrain_pred)
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
        
                list_yvalid_pred   = model(Xvalid, EmbValid)
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
                        #loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05, scaling = 0.5) 
                        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05) 
                        yvalid_true_ = torch.reshape(yvalid_true, (yvalid_true.shape[0], yvalid_true.shape[1]*yvalid_true.shape[2]*yvalid_true.shape[3], 1))
                        yvalid_pred  = torch.reshape(list_yvalid_pred[l], (list_yvalid_pred[l].shape[0], list_yvalid_pred[l].shape[1]*list_yvalid_pred[l].shape[2]*list_yvalid_pred[l].shape[3], 1))
                        valid_weight = torch.reshape(WgValid, (WgValid.shape[0], WgValid.shape[1]*WgValid.shape[2]*WgValid.shape[3], 1))
                        
                        #yvalid_true_ = yvalid_true[:, 0, :, :]
                        #yvalid_pred  = list_yvalid_pred[l][:, 0, :, :]
                        #valid_weight = WgValid[:, 0, :, :]
                        val_loss_w  = loss(valid_weight, yvalid_true_, valid_weight, yvalid_pred)
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
