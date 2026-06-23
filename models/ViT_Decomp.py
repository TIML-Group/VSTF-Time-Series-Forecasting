import sys
import torch
from torch import nn
from models.ViT_backbone import ViT_Backbone
from layers.Autoformer_EncDec import series_decomp_multi_moe

class Model(nn.Module):

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.pred_len = configs.pred_len
        self.decomp_module = series_decomp_multi_moe([7, 12, 14, 24, 48])
        self.model_trend = ViT_Backbone(configs)
        self.model_res = ViT_Backbone(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        res_init, trend_init = self.decomp_module(x_enc)
        res = self.model_res(res_init)
        trend = self.model_trend(trend_init)
        x = res + trend
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

