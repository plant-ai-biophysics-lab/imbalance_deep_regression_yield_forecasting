import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
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

# #PyTorch
# class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, targets, inputs, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
#
# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=0.0):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice
    

# from typing import List

# import torch
# import torch.nn.functional as F
# # from pytorch_toolbelt.utils.torch_utils import to_tensor
# from torch import Tensor
# from torch.nn.modules.loss import _Loss

# # from .functional import soft_dice_score

# __all__ = ["DiceLoss"]

# BINARY_MODE = "binary"
# MULTICLASS_MODE = "multiclass"
# MULTILABEL_MODE = "multilabel"

# def to_tensor(x, dtype=None) -> torch.Tensor:
#     if isinstance(x, torch.Tensor):
#         if dtype is not None:
#             x = x.type(dtype)
#         return x
#     if isinstance(x, np.ndarray) and x.dtype.kind not in {"O", "M", "U", "S"}:
#         x = torch.from_numpy(x)
#         if dtype is not None:
#             x = x.type(dtype)
#         return x
#     if isinstance(x, (list, tuple)):
#         x = np.ndarray(x)
#         x = torch.from_numpy(x)
#         if dtype is not None:
#             x = x.type(dtype)
#         return x

#     raise ValueError("Unsupported input type" + str(type(x)))

# def soft_dice_score(
#     output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
# ) -> torch.Tensor:
#     """

#     :param output:
#     :param target:
#     :param smooth:
#     :param eps:
#     :return:

#     Shape:
#         - Input: :math:`(N, NC, *)` where :math:`*` means any number
#             of additional dimensions
#         - Target: :math:`(N, NC, *)`, same shape as the input
#         - Output: scalar.

#     """
#     assert output.size() == target.size()
#     if dims is not None:
#         intersection = torch.sum(output * target, dim=dims)
#         cardinality = torch.sum(output + target, dim=dims)
#     else:
#         intersection = torch.sum(output * target)
#         cardinality = torch.sum(output + target)
#     dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
#     return dice_score


# class DiceLoss(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(
        self,
        mode: str = MULTILABEL_MODE,
        classes: List[int] = [1, 2, 3],
        log_loss=False,
        from_logits=True,
        smooth: float = 0.0,
        ignore_index=None,
        eps=1e-7,
    ):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()