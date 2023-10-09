import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import pdb
import math


device = "cuda" if torch.cuda.is_available() else "cpu"


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding = (kernel_size // 2), bias=bias, groups=groups)

class img_unfold(nn.Module):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    def __init__(self, config):
        super().__init__()
        self.kernel_size = config.kernel_size
        self.dilation    = config.dilation
        self.stride      = config.stride

        self.unfold = torch.nn.Unfold(kernel_size = self.kernel_size,
                            dilation    = self.dilation,
                            padding     = 0,
                            stride      = self.stride)
        

    def same_padding(self, x, ksizes = [3, 3], strides = [1, 1], rates = [1, 1]):
        assert len(x.size()) == 4
        batch_size, channel, rows, cols = x.size()
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        effective_k_row = (ksizes[0] - 1) * rates[0] + 1
        effective_k_col = (ksizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
        # Pad the input
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        x = torch.nn.ZeroPad2d(paddings)(x)
        return x
    
    def forward(self, x):
        x = self.same_padding(x)
        x = self.unfold(x)
        return x

class img_fold(nn.Module):

    def __init__(self, config):
        super().__init__()
        out_size    = (180, 360) 
        kernel_size = 1 #config.kernel_size
        dilation    = config.dilation
        stride      = config.stride

        self.fold = torch.nn.Fold(output_size=out_size,
                           kernel_size= kernel_size,
                           dilation   = dilation,
                           padding    = 0,
                           stride     = stride)

    def forward(self, x):

        x = self.fold(x)

        return x



class AuxEmb(nn.Module):

    def __init__(self, config):
        super(AuxEmb, self).__init__()

        n_patches  = 4

        self.aux_embeddings = TimestepEmbedder(config)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = Dropout(0.1)

    def forward(self, x):

    
        B = x[0].shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)


        cultivar_type = self.aux_embeddings(x[0])
        trellis_type  = self.aux_embeddings(x[1])
        row_space     = self.aux_embeddings(x[2])
        canopy_space  = self.aux_embeddings(x[3])


        aux = torch.cat([cultivar_type, trellis_type, row_space, canopy_space], dim = 1)
        # aux = torch.unsqueeze(aux, dim = 1)

        aux = torch.cat((cls_tokens, aux), dim = 1)

        embeddings = aux + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class SP_PatchEmbed(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    Input: x (size: TxWxHxC)==> 15x16x16x4
        
    number of patches = (image_size//patch_size)*(image_size//patch_size) = (16/8)*(16/8) = 4
    position encoding = 1 
    embedding size    = (8*8*4) * 15 (defult = 1024) ==> it can be any number dividable by the number of heads!

    Output: embeding token ((15*4 + 1 )x embeding size (1024)) = 1x61x1024

    """
    def __init__(self, config):
        super(SP_PatchEmbed, self).__init__()

        img_size   = _pair(16)
        patch_size = _pair(8)
        temporal_res = 15
        n_patches  = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * temporal_res # in_channels # 

        assert n_patches is not None, ('Number of Patches Can NOT be None!')

        self.patch_embeddings = Conv2d(in_channels    = config.in_channels,
                                       out_channels   = config.embed_dim,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = Dropout(0.1)

    def forward(self, x):

    
        B = x.shape[0]
        T = x.shape[-1]
        C = x.shape[1] 
        cls_tokens = self.cls_token.expand(B, -1, -1)


        x_stack = None
        for t in range(T): 
            this_time = x[:, :, :, :, t]
            this_time = self.patch_embeddings(this_time)

            this_time = this_time.flatten(2)
            this_time = this_time.transpose(-1, -2)

            if x_stack is None: 
                x_stack = this_time
            else:
                x_stack = torch.cat((x_stack, this_time), dim = 1)

        x_stack = torch.cat((cls_tokens, x_stack), dim = 1)
        embeddings = x_stack + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class img_aux_emb(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    Input: x (size: TxWxHxC)==> 15x16x16x4
        
    number of patches = (image_size//patch_size)*(image_size//patch_size) = (16/8)*(16/8) = 4
    position encoding = 1 
    embedding size    = (8*8*4) * 15 (defult = 1024) ==> it can be any number dividable by the number of heads!

    Output: embeding token ((15*4 + 1 )x embeding size (1024)) = 1x61x1024

    """
    def __init__(self, config):
        super(img_aux_emb, self).__init__()

        img_size   = _pair(16)
        patch_size = _pair(8)
        temporal_res = 15
        n_patches  = ((img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * temporal_res) + 4 # in_channels # 

        assert n_patches is not None, ('Number of Patches Can NOT be None!')

        self.patch_embeddings = Conv2d(in_channels    = config.in_channels,
                                       out_channels   = config.embed_dim,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = Dropout(0.1)

        self.aux_embeddings = TimestepEmbedder(config)

    def forward(self, x, aux):

        B = x.shape[0]
        T = x.shape[-1]
        C = x.shape[1] 
        cls_tokens = self.cls_token.expand(B, -1, -1)


        x_stack = None
        for t in range(T): 
            this_time = x[:, :, :, :, t]
            this_time = self.patch_embeddings(this_time)

            this_time = this_time.flatten(2)
            this_time = this_time.transpose(-1, -2)

            if x_stack is None: 
                x_stack = this_time
            else:
                x_stack = torch.cat((x_stack, this_time), dim = 1)


        cultivar_type = self.aux_embeddings(aux[0].to(device))
        trellis_type  = self.aux_embeddings(aux[1].to(device))
        row_space     = self.aux_embeddings(aux[2].to(device))
        canopy_space  = self.aux_embeddings(aux[3].to(device))

        aux = torch.cat([cultivar_type, trellis_type, row_space, canopy_space], dim = 1)

        x_stack = torch.cat((cls_tokens, x_stack, aux), dim = 1)

        embeddings = x_stack + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, config):
        super().__init__()
        self.proj  = nn.Conv2d(config.in_channels, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x):
 
        x = self.proj(x)
        # 1*180*360->  1024*10*20->1024*200->200*1024
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        return x

class PosEmbd(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, config):
        super().__init__()
        img_size                 = tuple((config.img_size, config.img_size))     # img_size = 180*360 
        patch_size               = tuple((config.patch_size, config.patch_size))    # patch_size = 18*18
        # num_patches              = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) # 10*20 = 200
        temporal_res = 15
        num_patches  = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * temporal_res # in_channels # 
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = nn.Dropout(config.Proj_drop)


    def forward(self, x):

        B  = x.shape[0] #
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        embeddings = x + self.position_embeddings 
        embeddings = self.dropout(embeddings)
        return embeddings

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.embed_dim, 4 * config.mlp_dim, bias=True)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.mlp_dim, config.embed_dim, bias=True)
        self.dropout = nn.Dropout(config.Proj_drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.c_fc.weight)
        nn.init.xavier_uniform_(self.c_proj.weight)
        nn.init.normal_(self.c_fc.bias, std=1e-6)
        nn.init.normal_(self.c_proj.bias, std=1e-6)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        # output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
        # regularization
        self.attn_dropout = nn.Dropout(config.Attn_drop)
        self.resid_dropout = nn.Dropout(config.Proj_drop)
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.dropout = config.Proj_drop
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.embed_dim, bias= True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.embed_dim, bias= True)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class UpHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # reshape 
        self.lfc      = nn.Linear(self.config.embed_dim, 4, bias=True) # 268
        self.fcn      = nn.GELU()
        self.fold     = torch.nn.Fold(output_size=(self.config.img_size, self.config.img_size),
                                      kernel_size= 1, dilation   = 1,
                                      padding    = 0, stride     = 1)


        # cnn + pixel shuffling  self.config.out_channels
        self.pixelshuffling       = nn.Sequential(default_conv(1, 1, 3, True),
                                      nn.PixelShuffle(1),
                                      nn.BatchNorm2d(1), #
                                      default_conv(1, 1, 3, True),
                                      nn.PixelShuffle(1),
                                      nn.BatchNorm2d(1), #LayerNorm((1, 720, 1440), bias= True), 
                                      )

    def forward(self, x):

        x = self.lfc(x)

        x = self.fcn(x)[:, 1:, :]


        x = x.view(x.shape[0], 1, int(self.config.img_size**2))

        x = self.fold(x)

        x = self.pixelshuffling(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, config, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, config.embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb  = self.mlp(t_freq)

        t_emb = torch.unsqueeze(t_emb, dim = 1)
        return t_emb
    

class ours(nn.Module):
    def __init__(self, config):
        super(ours, self).__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            AuxEmb       = AuxEmb(self.config),
            ImgAuxEmb    = img_aux_emb(self.config),
            PatchEmb     = SP_PatchEmbed(self.config),
            drop         = nn.Dropout(config.Proj_drop),
            h            = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f         = LayerNorm(config.embed_dim, bias= True),
            head         = UpHead(config),
        ))

    def forward(self, x, Mng_data):

        # x     = self.transformer.PatchEmb(x)       # token embeddings of shape (b, t, n_embd)
        # aux   = self.transformer.AuxEmb(Mng_data)
        # x = torch.cat([x, aux], dim = 1)

        x = self.transformer.ImgAuxEmb(x, Mng_data)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        x = self.transformer.head(x)
        return x

















