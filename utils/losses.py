import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import time
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


def mse_loss(inputs, targets):
    loss = (inputs - targets) ** 2
    loss = torch.mean(loss)
    
    return loss


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_integral_mse_loss(inputs, targets, weights=None):
    input_flat    = torch.flatten(inputs)
    target_flat   = torch.flatten(targets)
    weights_flat  = torch.flatten(weights)


    intervals = np.arange(0, 30)
    intervals = torch.as_tensor(intervals)
    ser = []
    trues, preds = [], []
    weights = []

    for phi in intervals:
        ytrue  = input_flat[torch.where((input_flat > phi) & (input_flat < phi +1))]  
        ypred  = target_flat[torch.where((input_flat > phi) & (input_flat < phi +1))]
        im_    = weights_flat[torch.where((input_flat > phi) & (input_flat < phi +1))]

        if len(ytrue) > 0: 
            #loss = (ytrue - ypred) ** 2
            #loss = loss * im_
            #loss = torch.trapz(torch.as_tensor(loss), dx=1) 
            #loss = torch.trapz(torch.as_tensor(loss), torch.as_tensor((phi, phi +1))) 
            #loss = torch.mean(loss) 
            #assert not torch.isnan(loss).any()
            true_mean = torch.sum(ytrue) 
            pred_mean = torch.sum(ypred) 
            #w_mean    = torch.mean(im_) 
        else: 
            X = 0.0
            #true_mean = 0.0
            #pred_mean = 0.0
            #w_mean    = 0.0
        
        #ser.append(loss)
        trues.append(true_mean)
        preds.append(pred_mean)
        #weights.append(w_mean)
    t = torch.as_tensor(trues)
    p = torch.as_tensor(preds)

    k = torch.argmax(torch.abs(t - p))
    loss = torch.abs(t[k] -p[k])
    #mul = torch.as_tensor(ser)*torch.as_tensor(weights)
    #loss = torch.sum((torch.as_tensor(trues)-torch.as_tensor(preds)) * torch.as_tensor(weights))
    #print(ser)
    #loss = torch.mean(torch.as_tensor(ser)) 
    #loss = torch.trapz(torch.as_tensor(ser), intervals) 
    return loss

def EMD(inputs, targets,):
    input_flat    = torch.flatten(inputs)
    target_flat   = torch.flatten(targets)

    intervals = np.arange(0, 30)
    intervals = torch.as_tensor(intervals)

    trues, preds = [], []

    for phi in intervals:
        ytrue  = input_flat[torch.where((input_flat > phi) & (input_flat < phi +1))]  
        ypred  = target_flat[torch.where((input_flat > phi) & (input_flat < phi +1))]

        if len(ytrue) > 0: 
            true_mean = torch.mean(ytrue) 
            pred_mean = torch.mean(ypred) 
        else: 
            true_mean = 0.0
            pred_mean = 0.0

        trues.append(true_mean)
        preds.append(pred_mean)

    t = torch.as_tensor(trues)
    P_t = t / torch.sum(P_t)


    p = torch.as_tensor(preds)
    P_r = p / torch.sum(P_r)


    l = 30.0
    D = torch.empty((l, l), dtype=torch.float32)

    for i in range(l):
        for j in range(l):
            D[i,j] = abs(range(l)[i] - range(l)[j])

    A_r = torch.zeros((l, l, l))
    A_t = torch.zeros((l, l, l))

    for i in range(l):
        for j in range(l):
            A_r[i, i, j] = 1
            A_t[i, j, i] = 1
            
    A = torch.cat((A_r.reshape((l, l**2)), A_t.reshape((l, l**2))), axis=0)


    b = torch.cat((P_r, P_t), axis=0)
    c = D.reshape((l**2))


    opt_res = torch.linalg.solve(-b, A.T, c)

    loss = -opt_res.fun


    return loss

def weighted_huber_mse_loss(inputs, targets, weights=None):

    input_flat    = torch.flatten(inputs)
    target_flat   = torch.flatten(targets)
    weights_flat  = torch.flatten(weights)


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

def custom_wasserstien_loss(ytrue_w, ytrue, ypred_w, ypred, blur: float, sinkhorn_nits: int, weighted_cost_func: True):


    # Compute the logarithm of the weights (needed in the softmin reduction) ---
    loga_i = torch.empty((ytrue_w.shape[0], ytrue_w.shape[1]), dtype = torch.float32).to(device) #requires_grad = True, 
    logb_j = torch.empty((ytrue_w.shape[0], ytrue_w.shape[1]), dtype = torch.float32).to(device) #requires_grad = True, 

    for b in range(ytrue_w.shape[0]):

        this_loga_i, this_logb_j = ytrue_w[b, :, 0].log(), ypred_w[b, :, 0].log()
        loga_i[b, :] = this_loga_i
        logb_j[b, :] = this_logb_j

    loga_i, logb_j = loga_i[:, :, None, None], logb_j[:, None, :, None]


    if weighted_cost_func is True: 

        D_w = (ytrue_w[:, :, :] @ (ypred_w[:, :, :]).transpose(1, 2))[:, :, :, None]
        C_ij = ((ytrue[:, :,None,:] - ypred[:, None,:,:]) ** 2).sum(-1) / 2
        C_ij = C_ij[:, :, :, None]  # reshape as a (N, M, 1) Tensor
        C_ij = C_ij*D_w
        C_ij.to(device)

    else: 
        C_ij = ((ytrue[:,None,:] - ypred[None,:,:]) ** 2).sum(-1) / 2
        C_ij = C_ij[:, :, None]  # reshape as a (N, M, 1) Tensor

    # Setup the dual variables -------------------------------------------------
    eps = blur ** 2  # "Temperature" epsilon associated to our blurring scale
    F_i, G_j = torch.zeros_like(loga_i, dtype = torch.float32), torch.zeros_like(
        logb_j, dtype = torch.float32
    )  # (scaled) dual vectors

    # Sinkhorn loop = coordinate ascent on the dual maximization problem -------
    for _ in range(sinkhorn_nits):
        F_i = -((-C_ij / eps + (G_j + logb_j))).logsumexp(dim=2)[:, :, None, :]
        G_j = -((-C_ij / eps + (F_i + loga_i))).logsumexp(dim=1)[:, None, :, :]

    # Return the dual vectors F and G, sampled on the x_i's and y_j's respectively:
    F_i, G_j = eps * F_i, eps * G_j

    list_batch_loss = torch.empty(ytrue.shape[0], dtype = torch.float32)
    for b in range(ytrue_w.shape[0]):
        # Returns the entropic transport cost associated to the dual variables F_i and G_j.''ArithmeticError
        entropic_transport_cost = ytrue_w[b, ...].view(-1).dot(F_i[b, ...].view(-1)) + ypred_w[b, ...].view(-1).dot(G_j[b, ...].view(-1))
        loss = (2 * entropic_transport_cost).sqrt()
        list_batch_loss[b] = loss
    
    return list_batch_loss

