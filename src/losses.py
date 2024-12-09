import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN

# import joblib
import time

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

class BMCLossMD(_Loss):
    """
    Multi-Dimension version BMC, compatible with 1-D BMC
    """

    def __init__(self, init_noise_sigma):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss

def bmc_loss_md(pred, target, noise_var):
    pred   = torch.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2]*pred.shape[3]))
    target = torch.reshape(target, (target.shape[0], target.shape[1]*target.shape[2]*target.shape[3]))

    I = torch.eye(pred.shape[-1], device="cuda")
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))
    loss = F.cross_entropy(logits, pred.cuda())
    loss = loss * (2 * noise_var).detach()
    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)

        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_vectorized(pred, target, noise_var)
        return loss.mean()

def bmc_loss_vectorized(pred, target, noise_var):
    # Adding an extra dimension to target for broadcasting
    target = target.unsqueeze(1)
    # Now logits calculation
    logits = -0.5 * (pred.unsqueeze(2) - target).pow(2) / noise_var
    # logits = -0.5 * (pred - target).pow(2) / noise_var
    labels = torch.arange(pred.shape[0], device=pred.device).expand(pred.shape[1], -1).T
    # loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction='none')
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none')

    loss = loss.view_as(pred) * (2 * noise_var).detach()
    return loss

class BNILoss(_Loss):
    def __init__(self, init_noise_sigma, bucket_centers, bucket_weights):
        super(BNILoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))
        self.bucket_centers = torch.tensor(bucket_centers).cuda()
        self.bucket_weights = torch.tensor(bucket_weights).cuda()

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bni_loss(pred, target, noise_var, self.bucket_centers, self.bucket_weights)
        return loss

def bni_loss(pred, target, noise_var, bucket_centers, bucket_weights):
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var

    num_bucket = bucket_centers.shape[0]
    bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
    bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

    balancing_term = - 0.5 * (pred.expand(-1, num_bucket) - bucket_center).pow(2) / noise_var + bucket_weights.log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()

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

