import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN
import time

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber_mse_loss(inputs, targets, weights=None):

    input_flat    = torch.flatten(inputs)
    target_flat   = torch.flatten(targets)
    weights_flat  = torch.flatten(weights)

    '''common_vales_true   = input_flat[torch.where((input_flat >= 8) & (input_flat < 19))]
    common_vales_pred   = target_flat[torch.where((input_flat >= 8) & (input_flat < 19))]
    
    low_extreme_values_true = input_flat[torch.where((input_flat > 0) & (input_flat < 8))]
    low_extreme_values_pred = target_flat[torch.where((input_flat > 0) & (input_flat < 8))]
    low_extreme_values_wg   = weights_flat[torch.where((input_flat > 0) & (input_flat <8))]

    high_extreme_values_true = input_flat[torch.where((input_flat >=19))]
    high_extreme_values_pred = target_flat[torch.where((input_flat >=19))]
    high_extreme_values_wg   = weights_flat[torch.where((input_flat >=19))]
    
        if len(low_extreme_values_true > 0) & len(high_extreme_values_true > 0):

        loss_low_extreme = (low_extreme_values_true - low_extreme_values_pred) ** 2
        loss_low_extreme *= low_extreme_values_wg.expand_as(loss_low_extreme)

        loss_high_extreme = (high_extreme_values_true - high_extreme_values_pred) ** 2
        loss_high_extreme *= high_extreme_values_wg.expand_as(loss_high_extreme)

        cat_matrix = torch.cat((loss_common, loss_low_extreme, loss_high_extreme))

        loss = torch.mean(cat_matrix) 
        
    elif len(low_extreme_values_true > 0) & len(high_extreme_values_true == 0): 
        loss_low_extreme = (low_extreme_values_true - low_extreme_values_pred) ** 2
        loss_low_extreme *= low_extreme_values_wg.expand_as(loss_low_extreme)

        cat_matrix = torch.cat((loss_common, loss_low_extreme))
        loss = torch.mean(cat_matrix)

    elif len(low_extreme_values_true == 0) & len(high_extreme_values_true > 0): 
        loss_high_extreme = (high_extreme_values_true - high_extreme_values_pred) ** 2
        loss_high_extreme *= high_extreme_values_wg.expand_as(loss_high_extreme)

        cat_matrix = torch.cat((loss_common, loss_high_extreme))
        loss = torch.mean(cat_matrix)

    else: 
        loss = torch.mean(loss_common)
    '''

    common_vales_true = input_flat[torch.where((weights_flat <= 1))]
    common_vales_pred = target_flat[torch.where((weights_flat <= 1))]

    extreme_vales_true = input_flat[torch.where((weights_flat > 1))]
    extreme_vales_pred = target_flat[torch.where((weights_flat > 1))]
    extreme_values_wg  = weights_flat[torch.where((weights_flat > 1))]

    loss_common   = (common_vales_true - common_vales_pred) ** 2

    loss_extreme = (extreme_vales_true - extreme_vales_pred) ** 2
    loss_extreme *= extreme_values_wg.expand_as(loss_extreme)

    cat_matrix = torch.cat((loss_common, loss_extreme))
    loss = torch.mean(cat_matrix)

    return loss



def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss






class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        training_start_time = time.time()
        pred   = torch.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2]*pred.shape[3]))
        target = torch.reshape(target, (target.shape[0], target.shape[1]*target.shape[2]*target.shape[3]))

        noise_var = self.noise_sigma ** 2
        losses = 0
        for i in range(pred.shape[1]): 
            pred_i = pred[:, i].unsqueeze(1)
            target_i = target[:, i].unsqueeze(1)

            loss = bmc_loss(pred_i, target_i, noise_var)
            losses += loss
        loss_mean = losses / pred.shape[1]
        #losses = [bmc_loss(pred[:, i].unsqueeze(1), target[:, i].unsqueeze(1), noise_var) for i in range(pred.shape[1])]
        #loss_mean = torch.mean(torch.Tensor(losses))

        #print(time.time() - training_start_time)  
        return loss_mean

 
def bmc_loss(pred, target, noise_var):
    
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()

    return loss

