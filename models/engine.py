import os
import time
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


device = "cuda" if torch.cuda.is_available() else "cpu"
from models.UNet2DConvLSTM import UNet2DConvLSTM
from src import losses
from models import configs

#======================================================================================================================================#
#====================================================== Training Config ===============================================================#
#======================================================================================================================================#   
seed = 1987 
torch.manual_seed(seed)
np.random.seed(seed)

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

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves a model checkpoint during training.

    Parameters:
    - state (dict): State to save, including model and optimizer states.
    - filename (str): File name to save the checkpoint.
    """
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads a checkpoint into a model and optimizer.

    Parameters:
    - checkpoint_path (str): Path to the checkpoint file.
    - model (nn.Module): Model to load the checkpoint into.
    - optimizer (torch.optim): Optimizer to load the checkpoint into.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_val_loss']

class YieldEst:
    """
    This class implements the training and evaluation of a neural network model for yield estimation.

    Attributes:
        model (nn.Module): The neural network model.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for regularization.
        exp (str): Experiment identifier used for file naming and directory structuring.
        optimizer (torch.optim): Optimizer for training the model.
        exp_output_dir (str): Directory for experiment outputs.
        best_model_name (str): Path for saving the best model.
        checkpoint_dir (str): Directory for saving checkpoints.
        loss_fig_name (str): File name for the loss figure.
        loss_df_name (str): File name for the loss data in CSV format.
        train_df_name (str): File name for the training data.
        valid_df_name (str): File name for the validation data.
        test_df_name (str): File name for the test data.
        timeseries_fig (str): File name for the time series figure.
        scatterplot (str): File name for the scatter plot.

    Methods:
        train: Trains the model using the provided data loaders.
        predict: Generates predictions using the trained model.
        _return_pred_df: Helper function to process prediction results into a DataFrame.
        xy_vector_generator: Generates X and Y coordinate vectors for a given point and window size.
    """

    def __init__(self, model, lr: float, wd: float, exp: str):
        # Initialize model parameters and directories
        self.model = model
        self.lr = lr
        self.wd = wd
        self.exp = exp

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)


        self.exp_output_dir = '/data2/hkaman/Projects/Imbalanced/EXPs/comp/' + 'EXP_' + self.exp

        self.best_model_name = os.path.join(self.exp_output_dir, 'best_model_' + self.exp + '.pth')
        self.last_model_name = os.path.join(self.exp_output_dir, 'last_model_' + self.exp + '.pth')
        self.best_checkpoint_dir = os.path.join(self.exp_output_dir, 'best_checkpoints_' + self.exp + '.pth')
        self.checkpoint_dir = os.path.join(self.exp_output_dir, 'checkpoints')
        self.loss_fig_name = os.path.join(self.exp_output_dir, 'loss', 'loss_' + self.exp + '.png')
        self.loss_df_name = os.path.join(self.exp_output_dir, 'loss', 'loss_' + self.exp + '.csv')
        self.train_df_name = os.path.join(self.exp_output_dir, self.exp + '_train.csv')
        self.valid_df_name = os.path.join(self.exp_output_dir, self.exp + '_valid.csv')
        self.test_df_name = os.path.join(self.exp_output_dir, self.exp + '_test.csv')
        self.timeseries_fig = os.path.join(self.exp_output_dir, self.exp + '_timeseries.png')
        self.scatterplot = os.path.join(self.exp_output_dir, self.exp + '_scatterplot.png')


    def train(self, 
              data_loader_training, 
              data_loader_validate, 
              loss: str, 
              epochs: int, 
              loss_stop_tolerance: int,
              reweighting_method: str):
        """
        Trains the model using the provided training and validation data loaders.

        Parameters:
        - data_loader_training: DataLoader for the training dataset.
        - data_loader_validate: DataLoader for the validation dataset.
        - loss_type (str): Type of loss function to use ('mse', 'wmse', 'huber', 'wass').
        - epochs (int): Number of epochs to train the model.
        - loss_stop_tolerance (int): Early stopping tolerance level.
        """

        best_val_loss  = 1e15 # initial dummy value
        early_stopping = EarlyStopping(tolerance = loss_stop_tolerance, min_delta = 50)
        loss_stats = {'train': [],"val": []}
        

        for epoch in range(1, epochs + 1):

            training_start_time = time.time()
            train_epoch_loss = 0
            self.model.train()

            for batch, sample in enumerate(data_loader_training):
                
                xtrain = sample['image'].to(device)
                ytrain_true = sample['mask'][:, :, :, :, 0].to(device)
                
                embmatrix_train = sample['EmbMatrix'].to(device)
                yieldzone_train = sample['YZ'].to(device)

                list_ytrain_pred  = self.model(xtrain, embmatrix_train, yieldzone_train)
                self.optimizer.zero_grad()

                if reweighting_method is None:
                    weight_train = None
                else:
                    weight_train = sample['weight'].to(device)

                train_loss = self._calculate_timeseries_loss(ytrain_true, list_ytrain_pred, weight_train, loss)

                train_loss.backward()
                self.optimizer.step()
                train_epoch_loss += train_loss.item() 

            # VALIDATION    
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_epoch_loss = 0
                for batch, sample in enumerate(data_loader_validate):
                    
                    xvalid = sample['image'].to(device)
                    yvalid_true = sample['mask'][:,:,:,:,0].to(device)
                    
                    embmatrix_valid = sample['EmbMatrix'].to(device)
                    yieldzone_valid = sample['YZ'].to(device)

                    list_yvalid_pred   = self.model(xvalid, embmatrix_valid, yieldzone_valid) 

                    if reweighting_method is None:
                        weight_valid = None
                    else:
                        weight_valid = sample['weight'].to(device)

                    valid_loss = self._calculate_timeseries_loss(yvalid_true, list_yvalid_pred, weight_valid, loss)

                    val_epoch_loss += valid_loss.item()

            loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
            loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))

            training_duration_time = (time.time() - training_start_time)        
            print(f'Epoch {epoch+0:03}: | Time(s): {training_duration_time:.3f}| Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 

            checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss
            }

            save_checkpoint(checkpoint, filename= os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

            if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                        
                best_val_loss=(val_epoch_loss/len(data_loader_validate))
                torch.save(self.model.state_dict(), self.best_model_name)

                save_checkpoint(checkpoint, filename = self.best_checkpoint_dir)

                # early_stopping.update(False)
                print(f'=============================== Best model Saved! Val MSE: {best_val_loss:.4f}')

                status = True
            else:

                status = False

            early_stopping(status)
            if early_stopping.early_stop:
                print("Early stopping triggered at epoch:", epoch)
                torch.save(self.model.state_dict(), self.last_model_name)
                break

        save_loss_df(loss_stats, self.loss_df_name, self.loss_fig_name)

    def _calculate_timeseries_loss(self, y_true, list_y_pred, weight, loss_type):
        """
        Calculates the cumulative mean squared error loss for a list of predictions or a single prediction.

        Parameters:
        - y_true (Tensor): The ground truth values.
        - list_y_pred (list of Tensors or Tensor): List of predicted values or a single tensor.
        - weight (Tensor): The weight for weighted loss calculations.
        - loss_type (str): The type of loss ('mse', 'wmse', 'huber').

        Returns:
        - Tensor: The cumulative MSE loss for all predictions.
        """

        # Check if list_y_pred is a list or a single tensor

        if isinstance(list_y_pred, list):
            total_loss = 0.0
            for y_pred in list_y_pred:
                if loss_type == 'mse':
                    current_loss = losses.mse_loss(y_pred, y_true)
                elif loss_type == 'wmse':
                    current_loss = losses.weighted_mse_loss(y_pred, y_true, weight)
                elif loss_type == 'huber':
                    current_loss = losses.weighted_huber_mse_loss(y_pred, y_true, weight)
                elif loss_type == 'focal':
                    current_loss = losses.weighted_focal_mse_loss(y_pred, y_true, weights=None, activate='sigmoid', beta=.2, gamma=1)
                elif loss_type == 'focal-r':
                    current_loss = losses.weighted_focal_l1_loss(y_pred, y_true, weights=weight, activate='sigmoid', beta=.2, gamma=1)
                # elif loss_type == 'wass':
                #     current_loss = SamplesLoss(y_pred, y_true)
                elif loss_type =='bmc':
                    current_loss = losses.BMCLoss(init_noise_sigma = 0.5)(y_pred, y_true)
                
                total_loss += current_loss
        else:
            # list_y_pred is a single tensor
            y_pred = list_y_pred
            if loss_type == 'mse':
                total_loss = losses.mse_loss(y_pred, y_true) #
            elif loss_type == 'wmse':
                total_loss = losses.weighted_mse_loss(y_pred, y_true, weight)
            elif loss_type == 'huber':
                total_loss = losses.weighted_huber_mse_loss(y_pred, y_true, weight)
            elif loss_type == 'focal':
                total_loss = losses.weighted_focal_mse_loss(y_pred, y_true, weights=None, activate='sigmoid', beta=.2, gamma=1)
            elif loss_type =='bmc':
                total_loss = losses.BMCLoss(init_noise_sigma = 8.0)(y_pred, y_true)
                

        return total_loss

    def predict(self, config, data_loader, category: str, iter: int):

        print(f"*************** No YZ Strategy / Eval Process! **************")
        model = UNet2DConvLSTM(config, cond = False).to(device)
        model.load_state_dict(torch.load(self.best_model_name))
        output_files =[]

        for i in range(iter):
            with torch.no_grad():
                for sample in data_loader:
                    x = sample['image'].to(device)
                    y = sample['mask'].detach().cpu().numpy()
                    emblist = sample['EmbList']
                    block_id = sample['block']
                    block_cultivar_id = sample['cultivar']
                    block_x_coords = sample['X']
                    block_y_coords = sample['Y']
                    embmatrix = sample['EmbMatrix'].to(device)
                    yieldzone = sample['YZ'].to(device)
                
                    pred_list = self.model(x, embmatrix, yieldzone)

                    this_batch = {"block": block_id, 
                                        "cultivar": block_cultivar_id, 
                                        "X": block_x_coords, "Y": block_y_coords,
                                        "ytrue": y}
                    for i, pred in enumerate(pred_list):
                        key = f"ypred_w{i+1}"  # Creates keys like ypred_w1, ypred_w2, ..., ypred_wN
                        this_batch[key] = pred.detach().cpu().numpy()


                    output_files.append(this_batch)

                modified_df = self._return_modified_pred_df(output_files, None, 16)
                if category == 'train':
                    name_tr = self.train_df_name[:-4]  + '.csv'
                    modified_df.to_csv(name_tr)
                    print("train inference is done!")
                elif category == 'valid':
                    name_val = self.valid_df_name[:-4]  + '.csv'
                    modified_df.to_csv(name_val)
                    print("validation inference is done!")
                elif category == 'test':
                    name_te = self.test_df_name[:-4] + '.csv'
                    modified_df.to_csv(name_te)
                    print("test inference is done!")

    def _return_modified_pred_df(self, pred_npy, blocks_list, wsize=None):

        if blocks_list is None: 
            all_block_names = [dict['block'] for dict in pred_npy]#[0]
            blocks_list = list(set(item for sublist in all_block_names for item in sublist))


        OutDF = pd.DataFrame()
        columns = ['block', 'cultivar', 'x', 'y', 'ytrue']
        data = {col: [] for col in columns}  # Initialize dictionary for DataFrame

        # Initialize lists for predictions dynamically based on the first item's keys
        pred_keys = [key for key in pred_npy[0].keys() if key.startswith('ypred')]
        for key in pred_keys:
            data[key] = []

        for block in blocks_list:
            name_split = os.path.split(block)[-1]
            block_name = name_split.replace(name_split[7:], '')
            root_name = name_split.replace(name_split[:4], '').replace(name_split[3], '')
            block_id = root_name
            
            res = {key: configs.blocks_information[key] for key in configs.blocks_information.keys() & {block_name}}
            list_d = res.get(block_name)
            cultivar_id = list_d[1]
        
            for l in range(len(pred_npy)):
                tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
                if len(tb_pred_indices) !=0:   
                    for index in tb_pred_indices:

                        x0 = pred_npy[l]['X'][index]
                        y0 = pred_npy[l]['Y'][index]
                        x_vector, y_vector = self.xy_vector_generator(x0, y0, wsize)
                        data['x'].append(x_vector)
                        data['y'].append(y_vector)
                        data['ytrue'].append(pred_npy[l]['ytrue'][index].flatten())

                        tb_block_id = np.array(len(pred_npy[l]['ytrue'][index].flatten())*[block_id], dtype=np.int32)
                        data['block'].append(tb_block_id)

                        tb_cultivar_id = np.array(len(pred_npy[l]['ytrue'][index].flatten())*[cultivar_id], dtype=np.int8)
                        data['cultivar'].append(tb_cultivar_id)



                        # Handle predictions dynamically
                        for key in pred_keys:
                            flattened_pred = pred_npy[l][key][index].flatten()
                            data[key].append(flattened_pred)

        empty_dict = {key: None for key in data.keys()}
        # Convert lists to numpy arrays for consistency
        for key in data:
            if data[key]:  # Ensure there's data to concatenate
                # print(len(data[key]))
                output = np.concatenate(data[key])
                empty_dict[key] = output
                # print(key, output.shape)

        # Create DataFrame from data dictionary
        OutDF = pd.DataFrame(empty_dict)
        return OutDF

    
    def xy_vector_generator(self, x0, y0, wsize):


        x_vector, y_vector = [], []
        
        for i in range(x0, x0+wsize):
            for j in range(y0, y0+wsize):
                x_vector.append(i)
                y_vector.append(j)

        return x_vector, y_vector 
    




