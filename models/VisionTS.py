import sys
import einops
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import torch.nn.functional as F
import timm

import inspect
from PIL import Image
from torchvision.transforms import Resize
from . import models_mae


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

def safe_resize(size, interpolation):
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.padding = stride
        self.patch_len = patch_len
        self.stride = stride

        self.pad_left = 0
        self.pad_right = 0
        self.periodicity = 1
        self.norm_const = 0.4
        align_const = 0.4
        self.vision_model = models_mae.mae_vit_base_patch16()
        checkpoint = torch.load("/u/jliu61/TS_Research/mae_visualize_vit_base.pth", map_location='cpu')
        self.vision_model.load_state_dict(checkpoint['model'], strict=True)
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size
        input_ratio = (self.pad_left + self.seq_len) / (
                    self.pad_left + self.seq_len + self.pad_right + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        adjust_input_ratio = self.num_patch_input / self.num_patch

        self.input_resize = safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)),
                                             interpolation=Image.BILINEAR)
        self.scale_x = ((self.pad_left + self.seq_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.num_patch_output = self.num_patch - self.num_patch_input

        self.output_resize = safe_resize((1, int(round(self.image_size * self.scale_x))), interpolation=Image.BILINEAR)
        self.norm_const = 0.4
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev
        # Channel Independent
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')  # [bs x nvars x seq_len]

        # 2. Segmentation
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate')  # [b n s]
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)
        # 3. Render & Alignment
        x_resize = self.input_resize(x_2d)
        
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

        # 4. Reconstruction
        _, y, mask = self.vision_model(
            image_input, 
            mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
        )
        image_reconstructed = self.vision_model.unpatchify(y) # [(bs x nvars) x 3 x h x w]
        
        # 5. Forecasting
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # color image to grey
        y_segmentations = self.output_resize(y_grey) # resize back
        y_flatten = einops.rearrange(
            y_segmentations, 
            '(b n) 1 f p -> b (p f) n', 
            b=x_enc.shape[0], f=self.periodicity
        ) # flatten
        y = y_flatten[:, self.pad_left + self.seq_len: self.pad_left + self.seq_len + self.pred_len, :] # extract the forecasting window

        # 6. Denormalization
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
