import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN
import time
import pandas as pd
from scipy.integrate import simps
from sklearn.metrics import make_scorer
import numpy as np
from geomloss import SamplesLoss
device = "cuda" if torch.cuda.is_available() else "cpu"
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


'''def custom_wasserstien_loss(ytrue_w, ytrue, ypred_w, ypred, blur: float, sinkhorn_nits: int, weighted_cost_func: True):

    list_batch_loss = torch.empty(ytrue.shape[0], dtype = torch.float32) #requires_grad = True, 
    for b in range(ytrue.shape[0]):

        loga_i, logb_j = ytrue_w[b, :, 0].log(), ypred_w[b, :, 0].log()
        loga_i, logb_j = loga_i[:, None, None], logb_j[None, :, None]

        if weighted_cost_func is True: 
            D_w = (ytrue_w[b, :, :] @ (ypred_w[b, :, :]).t())[:, :, None]
            C_ij = ((ytrue[b, :, None, :] - ypred[b, None,:,:]) ** 2).sum(-1) / 2
            C_ij = C_ij[:, :, None]  # reshape as a (N, M, 1) Tensor
            C_ij = C_ij*D_w

        else: 
            C_ij = ((ytrue[b, :,None,:] - ypred[b, None,:,:]) ** 2).sum(-1) / 2
            C_ij = C_ij[:, :, None]  # reshape as a (N, M, 1) Tensor

        # Setup the dual variables -------------------------------------------------
        eps = blur ** 2  # "Temperature" epsilon associated to our blurring scale
        F_i, G_j = torch.zeros_like(loga_i, dtype = torch.float32), torch.zeros_like(
            logb_j, dtype = torch.float32
        )  # (scaled) dual vectors

        # Sinkhorn loop = coordinate ascent on the dual maximization problem -------
        for _ in range(sinkhorn_nits):
            F_i = -((-C_ij / eps + (G_j + logb_j))).logsumexp(dim=1)[:, None, :]
            G_j = -((-C_ij / eps + (F_i + loga_i))).logsumexp(dim=0)[None, :, :]

        # Return the dual vectors F and G, sampled on the x_i's and y_j's respectively:
        F_i, G_j = eps * F_i, eps * G_j
        F_i, G_j = F_i.view(-1), G_j.view(-1)
        
        # Returns the entropic transport cost associated to the dual variables F_i and G_j.''ArithmeticError
        entropic_transport_cost = ytrue_w[b, :, 0].dot(F_i) + ypred_w[b, :, 0].dot(G_j)
        loss = (2 * entropic_transport_cost).sqrt()

        list_batch_loss[b] = loss

    return list_batch_loss'''

'''
def calc_sera(y_true, y_pred,x_relevance=None):
    # creating a list from 0 to 1 with 0.001 interval
    start_range = 0
    end_range = 1
    interval_size = 0.001

    list_1 = [round(val * interval_size, 3) for val in range(1, 1000)]
    list_1.append(start_range)
    list_1.append(end_range)
    epsilon = sorted(list_1, key=lambda x: float(x))
    
    df = pd.concat([y_true,y_pred,x_relevance],axis=1,keys= ['true', 'pred', 'phi'])
    # Initiating lists to store relevance(phi) and squared-error relevance (ser)
    relevance = []
    ser = []

    # Converting the dataframe to a numpy array
    rel_arr = x_relevance
    # selecting a phi value
    for phi in epsilon:
        relevance.append(phi)
        error_squared_sum = 0
        error_squared_sum = sum((df[df.phi>=phi]['true'] - df[df.phi>=phi]['pred'])**2)
        ser.append(error_squared_sum)

    # squared-error relevance area (sera)
#     numerical integration using simps(y, x)

    sera = simps(ser, relevance)

    return sera

sera = make_scorer(calc_sera, x_relevance=X['relevance'], greater_is_better=False)'''


'''
import torch
from torch.nn import Module
from functools import partial
import warnings

from .kernel_samples import kernel_tensorized, kernel_online, kernel_multiscale

from .sinkhorn_samples import sinkhorn_tensorized
from .sinkhorn_samples import sinkhorn_online
from .sinkhorn_samples import sinkhorn_multiscale

from .kernel_samples import kernel_tensorized as hausdorff_tensorized
from .kernel_samples import kernel_online as hausdorff_online
from .kernel_samples import kernel_multiscale as hausdorff_multiscale


routines = {
    "sinkhorn": {
        "tensorized": sinkhorn_tensorized,
        "online": sinkhorn_online,
        "multiscale": sinkhorn_multiscale,
    },
    "hausdorff": {
        "tensorized": hausdorff_tensorized,
        "online": hausdorff_online,
        "multiscale": hausdorff_multiscale,
    },
    "energy": {
        "tensorized": partial(kernel_tensorized, name="energy"),
        "online": partial(kernel_online, name="energy"),
        "multiscale": partial(kernel_multiscale, name="energy"),
    },
    "gaussian": {
        "tensorized": partial(kernel_tensorized, name="gaussian"),
        "online": partial(kernel_online, name="gaussian"),
        "multiscale": partial(kernel_multiscale, name="gaussian"),
    },
    "laplacian": {
        "tensorized": partial(kernel_tensorized, name="laplacian"),
        "online": partial(kernel_online, name="laplacian"),
        "multiscale": partial(kernel_multiscale, name="laplacian"),
    },
}


[docs]class SamplesLoss(Module):
    """Creates a criterion that computes distances between sampled measures on a vector space.

    Warning:
        If **loss** is ``"sinkhorn"`` and **reach** is **None** (balanced Optimal Transport),
        the resulting routine will expect measures whose total masses are equal with each other.

    Parameters:
        loss (string, default = ``"sinkhorn"``): The loss function to compute.
            The supported values are:

              - ``"sinkhorn"``: (Un-biased) Sinkhorn divergence, which interpolates
                between Wasserstein (blur=0) and kernel (blur= :math:`+\infty` ) distances.
              - ``"hausdorff"``: Weighted Hausdorff distance, which interpolates
                between the ICP loss (blur=0) and a kernel distance (blur= :math:`+\infty` ).
              - ``"energy"``: Energy Distance MMD, computed using the kernel
                :math:`k(x,y) = -\|x-y\|_2`.
              - ``"gaussian"``: Gaussian MMD, computed using the kernel
                :math:`k(x,y) = \exp \\big( -\|x-y\|_2^2 \,/\, 2\sigma^2)`
                of standard deviation :math:`\sigma` = **blur**.
              - ``"laplacian"``: Laplacian MMD, computed using the kernel
                :math:`k(x,y) = \exp \\big( -\|x-y\|_2 \,/\, \sigma)`
                of standard deviation :math:`\sigma` = **blur**.

        p (int, default=2): If **loss** is ``"sinkhorn"`` or ``"hausdorff"``,
            specifies the ground cost function between points.
            The supported values are:

              - **p** = 1: :math:`~~C(x,y) ~=~ \|x-y\|_2`.
              - **p** = 2: :math:`~~C(x,y) ~=~ \\tfrac{1}{2}\|x-y\|_2^2`.

        blur (float, default=.05): The finest level of detail that
            should be handled by the loss function - in
            order to prevent overfitting on the samples' locations.

            - If **loss** is ``"gaussian"`` or ``"laplacian"``,
              it is the standard deviation :math:`\sigma` of the convolution kernel.
            - If **loss** is ``"sinkhorn"`` or ``"hausdorff"``,
              it is the typical scale :math:`\sigma` associated
              to the temperature :math:`\\varepsilon = \sigma^p`.
              The default value of .05 is sensible for input
              measures that lie in the unit square/cube.

            Note that the *Energy Distance* is scale-equivariant, and won't
            be affected by this parameter.

        reach (float, default=None= :math:`+\infty` ): If **loss** is ``"sinkhorn"``
            or ``"hausdorff"``,
            specifies the typical scale :math:`\\tau` associated
            to the constraint strength :math:`\\rho = \\tau^p`.

        diameter (float, default=None): A rough indication of the maximum
            distance between points, which is used to tune the :math:`\\varepsilon`-scaling
            descent and provide a default heuristic for clustering **multiscale** schemes.
            If **None**, a conservative estimate will be computed on-the-fly.

        scaling (float, default=.5): If **loss** is ``"sinkhorn"``,
            specifies the ratio between successive values
            of :math:`\sigma=\\varepsilon^{1/p}` in the
            :math:`\\varepsilon`-scaling descent.
            This parameter allows you to specify the trade-off between
            speed (**scaling** < .4) and accuracy (**scaling** > .9).

        truncate (float, default=None= :math:`+\infty`): If **backend**
            is ``"multiscale"``, specifies the effective support of
            a Gaussian/Laplacian kernel as a multiple of its standard deviation.
            If **truncate** is not **None**, kernel truncation
            steps will assume that
            :math:`\\exp(-x/\sigma)` or
            :math:`\\exp(-x^2/2\sigma^2) are zero when
            :math:`\|x\| \,>\, \\text{truncate}\cdot \sigma`.


        cost (function or string, default=None): if **loss** is ``"sinkhorn"``
            or ``"hausdorff"``, specifies the cost function that should
            be used instead of :math:`\\tfrac{1}{p}\|x-y\|^p`:

            - If **backend** is ``"tensorized"``, **cost** should be a
              python function that takes as input a
              (B,N,D) torch Tensor **x**, a (B,M,D) torch Tensor **y**
              and returns a batched Cost matrix as a (B,N,M) Tensor.
            - Otherwise, if **backend** is ``"online"`` or ``"multiscale"``,
              **cost** should be a `KeOps formula <http://www.kernel-operations.io/api/math-operations.html>`_,
              given as a string, with variables ``X`` and ``Y``.
              The default values are ``"Norm2(X-Y)"`` (for **p** = 1) and
              ``"(SqDist(X,Y) / IntCst(2))"`` (for **p** = 2).

        cluster_scale (float, default=None): If **backend** is ``"multiscale"``,
            specifies the coarse scale at which cluster centroids will be computed.
            If **None**, a conservative estimate will be computed from
            **diameter** and the ambient space's dimension,
            making sure that memory overflows won't take place.

        debias (bool, default=True): If **loss** is ``"sinkhorn"``,
            specifies if we should compute the **unbiased**
            Sinkhorn divergence instead of the classic,
            entropy-regularized "SoftAssign" loss.

        potentials (bool, default=False): When this parameter is set to True,
            the :mod:`SamplesLoss` layer returns a pair of optimal dual potentials
            :math:`F` and :math:`G`, sampled on the input measures,
            instead of differentiable scalar value.
            These dual vectors :math:`(F(x_i))` and :math:`(G(y_j))`
            are encoded as Torch tensors, with the same shape
            as the input weights :math:`(\\alpha_i)` and :math:`(\\beta_j)`.

        verbose (bool, default=False): If **backend** is ``"multiscale"``,
            specifies whether information on the clustering and
            :math:`\\varepsilon`-scaling descent should be displayed
            in the standard output.

        backend (string, default = ``"auto"``): The implementation that
            will be used in the background; this choice has a major impact
            on performance. The supported values are:

              - ``"auto"``: Choose automatically, using a simple
                heuristic based on the inputs' shapes.
              - ``"tensorized"``: Relies on a full cost/kernel matrix, computed
                once and for all and stored on the device memory.
                This method is fast, but has a quadratic
                memory footprint and does not scale beyond ~5,000 samples per measure.
              - ``"online"``: Computes cost/kernel values on-the-fly, leveraging
                online map-reduce CUDA routines provided by
                the `pykeops <https://www.kernel-operations.io>`_ library.
              - ``"multiscale"``: Fast implementation that scales to millions
                of samples in dimension 1-2-3, relying on the block-sparse
                reductions provided by the `pykeops <https://www.kernel-operations.io>`_ library.

    """

    def __init__(
        self,
        loss="sinkhorn",
        p=2,
        blur=0.05,
        reach=None,
        diameter=None,
        scaling=0.5,
        truncate=5,
        cost=None,
        kernel=None,
        cluster_scale=None,
        debias=True,
        potentials=False,
        verbose=False,
        backend="auto",
    ):

        super(SamplesLoss, self).__init__()
        self.loss = loss
        self.backend = backend
        self.p = p
        self.blur = blur
        self.reach = reach
        self.truncate = truncate
        self.diameter = diameter
        self.scaling = scaling
        self.cost = cost
        self.kernel = kernel
        self.cluster_scale = cluster_scale
        self.debias = debias
        self.potentials = potentials
        self.verbose = verbose

    def forward(self, *args):
        """Computes the loss between sampled measures.

        Documentation and examples: Soon!
        Until then, please check the tutorials :-)"""

        l_x, α, x, l_y, β, y = self.process_args(*args)
        B, N, M, D, l_x, α, l_y, β = self.check_shapes(l_x, α, x, l_y, β, y)

        backend = (
            self.backend
        )  # Choose the backend -----------------------------------------
        if l_x is not None or l_y is not None:
            if backend in ["auto", "multiscale"]:
                backend = "multiscale"
            else:
                raise ValueError(
                    'Explicit cluster labels are only supported with the "auto" and "multiscale" backends.'
                )

        elif backend == "auto":
            if M * N <= 5000 ** 2:
                backend = (
                    "tensorized"  # Fast backend, with a quadratic memory footprint
                )
            else:
                if (
                    D <= 3
                    and self.loss == "sinkhorn"
                    and M * N > 10000 ** 2
                    and self.p == 2
                ):
                    backend = "multiscale"  # Super scalable algorithm in low dimension
                else:
                    backend = "online"  # Play it safe, without kernel truncation

        # Check compatibility between the batchsize and the backend --------------------------

        if backend in ["multiscale"]:  # multiscale routines work on single measures
            if B == 1:
                α, x, β, y = α.squeeze(0), x.squeeze(0), β.squeeze(0), y.squeeze(0)
            elif B > 1:
                warnings.warn(
                    "The 'multiscale' backend do not support batchsize > 1. "
                    + "Using 'tensorized' instead: beware of memory overflows!"
                )
                backend = "tensorized"

        if B == 0 and backend in [
            "tensorized",
            "online",
        ]:  # tensorized and online routines work on batched tensors
            α, x, β, y = α.unsqueeze(0), x.unsqueeze(0), β.unsqueeze(0), y.unsqueeze(0)

        # Run --------------------------------------------------------------------------------
        values = routines[self.loss][backend](
            α,
            x,
            β,
            y,
            p=self.p,
            blur=self.blur,
            reach=self.reach,
            diameter=self.diameter,
            scaling=self.scaling,
            truncate=self.truncate,
            cost=self.cost,
            kernel=self.kernel,
            cluster_scale=self.cluster_scale,
            debias=self.debias,
            potentials=self.potentials,
            labels_x=l_x,
            labels_y=l_y,
            verbose=self.verbose,
        )

        # Make sure that the output has the correct shape ------------------------------------
        if (
            self.potentials
        ):  # Return some dual potentials (= test functions) sampled on the input measures
            F, G = values
            return F.view_as(α), G.view_as(β)

        else:  # Return a scalar cost value
            if backend in ["multiscale"]:  # KeOps backends return a single scalar value
                if B == 0:
                    return values  # The user expects a scalar value
                else:
                    return values.view(
                        -1
                    )  # The user expects a "batch list" of distances

            else:  # "tensorized" backend returns a "batch vector" of values
                if B == 0:
                    return values[0]  # The user expects a scalar value
                else:
                    return values  # The user expects a "batch vector" of distances

    def process_args(self, *args):
        if len(args) == 6:
            return args
        if len(args) == 4:
            α, x, β, y = args
            return None, α, x, None, β, y
        elif len(args) == 2:
            x, y = args
            α = self.generate_weights(x)
            β = self.generate_weights(y)
            return None, α, x, None, β, y
        else:
            raise ValueError(
                "A SamplesLoss accepts two (x, y), four (α, x, β, y) or six (l_x, α, x, l_y, β, y)  arguments."
            )

    def generate_weights(self, x):
        if x.dim() == 2:  #
            N = x.shape[0]
            return torch.ones(N).type_as(x) / N
        elif x.dim() == 3:
            B, N, _ = x.shape
            return torch.ones(B, N).type_as(x) / N
        else:
            raise ValueError(
                "Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors."
            )

    def check_shapes(self, l_x, α, x, l_y, β, y):

        if α.dim() != β.dim():
            raise ValueError(
                "Input weights 'α' and 'β' should have the same number of dimensions."
            )
        if x.dim() != y.dim():
            raise ValueError(
                "Input samples 'x' and 'y' should have the same number of dimensions."
            )
        if x.shape[-1] != y.shape[-1]:
            raise ValueError(
                "Input samples 'x' and 'y' should have the same last dimension."
            )

        if (
            x.dim() == 2
        ):  # No batch --------------------------------------------------------------------
            B = 0  # Batchsize
            N, D = x.shape  # Number of "i" samples, dimension of the feature space
            M, _ = y.shape  # Number of "j" samples, dimension of the feature space

            if α.dim() not in [1, 2]:
                raise ValueError(
                    "Without batches, input weights 'α' and 'β' should be encoded as (N,) or (N,1) tensors."
                )
            elif α.dim() == 2:
                if α.shape[1] > 1:
                    raise ValueError(
                        "Without batches, input weights 'α' should be encoded as (N,) or (N,1) tensors."
                    )
                if β.shape[1] > 1:
                    raise ValueError(
                        "Without batches, input weights 'β' should be encoded as (M,) or (M,1) tensors."
                    )
                α, β = α.view(-1), β.view(-1)

            if l_x is not None:
                if l_x.dim() not in [1, 2]:
                    raise ValueError(
                        "Without batches, the vector of labels 'l_x' should be encoded as an (N,) or (N,1) tensor."
                    )
                elif l_x.dim() == 2:
                    if l_x.shape[1] > 1:
                        raise ValueError(
                            "Without batches, the vector of labels 'l_x' should be encoded as (N,) or (N,1) tensors."
                        )
                    l_x = l_x.view(-1)
                if len(l_x) != N:
                    raise ValueError(
                        "The vector of labels 'l_x' should have the same length as the point cloud 'x'."
                    )

            if l_y is not None:
                if l_y.dim() not in [1, 2]:
                    raise ValueError(
                        "Without batches, the vector of labels 'l_y' should be encoded as an (M,) or (M,1) tensor."
                    )
                elif l_y.dim() == 2:
                    if l_y.shape[1] > 1:
                        raise ValueError(
                            "Without batches, the vector of labels 'l_y' should be encoded as (M,) or (M,1) tensors."
                        )
                    l_y = l_y.view(-1)
                if len(l_y) != M:
                    raise ValueError(
                        "The vector of labels 'l_y' should have the same length as the point cloud 'y'."
                    )

            N2, M2 = α.shape[0], β.shape[0]

        elif (
            x.dim() == 3
        ):  # batch computation ---------------------------------------------------------
            (
                B,
                N,
                D,
            ) = x.shape
            # Batchsize, number of "i" samples, dimension of the feature space
            (
                B2,
                M,
                _,
            ) = y.shape
            # Batchsize, number of "j" samples, dimension of the feature space
            if B != B2:
                raise ValueError("Samples 'x' and 'y' should have the same batchsize.")

            if α.dim() not in [2, 3]:
                raise ValueError(
                    "With batches, input weights 'α' and 'β' should be encoded as (B,N) or (B,N,1) tensors."
                )
            elif α.dim() == 3:
                if α.shape[2] > 1:
                    raise ValueError(
                        "With batches, input weights 'α' should be encoded as (B,N) or (B,N,1) tensors."
                    )
                if β.shape[2] > 1:
                    raise ValueError(
                        "With batches, input weights 'β' should be encoded as (B,M) or (B,M,1) tensors."
                    )
                α, β = α.squeeze(-1), β.squeeze(-1)

            if l_x is not None:
                raise NotImplementedError(
                    'The "multiscale" backend has not been implemented with batches.'
                )
            if l_y is not None:
                raise NotImplementedError(
                    'The "multiscale" backend has not been implemented with batches.'
                )

            B2, N2 = α.shape
            B3, M2 = β.shape
            if B != B2:
                raise ValueError(
                    "Samples 'x' and weights 'α' should have the same batchsize."
                )
            if B != B3:
                raise ValueError(
                    "Samples 'y' and weights 'β' should have the same batchsize."
                )

        else:
            raise ValueError(
                "Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors."
            )

        if N != N2:
            raise ValueError(
                "Weights 'α' and samples 'x' should have compatible shapes."
            )
        if M != M2:
            raise ValueError(
                "Weights 'β' and samples 'y' should have compatible shapes."
            )

        return B, N, M, D, l_x, α, l_y, β'''