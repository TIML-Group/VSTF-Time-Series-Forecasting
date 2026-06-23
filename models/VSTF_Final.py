import torch
from torch import nn
import torch.fft
import timm
import math

# Imports as per the original files
from layers.Embed import PatchEmbedding
from layers.TSI_Encoder import *

# -----------------------------------------------------------------------------
# 1. Helper Modules (Shared & Decomposition)
# -----------------------------------------------------------------------------

def build_tsti_encoder(configs, patch_num, patch_len):
    encoder = TSTiEncoder(configs.enc_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=2048,
                            n_layers=3, d_model=configs.d_model, n_heads=configs.n_heads, d_k=None, d_v=None, d_ff=configs.d_ff,
                            attn_dropout=0, dropout=configs.dropout, act="gelu", key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                               pe='zeros', learn_pe=True, verbose=False)
    return encoder

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = nn.ModuleList([series_decomp(kernel) for kernel in kernel_size])

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class SeriesDecompPolynomial(nn.Module):
    """
    Decomposition via Polynomial Regression.
    """
    def __init__(self, seq_len, poly_degree=1):
        super().__init__()
        self.seq_len = seq_len
        self.poly_degree = poly_degree
        
        # 1. Create the design matrix T: [seq_len, degree+1]
        t = torch.arange(seq_len, dtype=torch.float32)
        T = torch.stack([t ** i for i in range(poly_degree + 1)], dim=1) # [L, D+1]
        
        # 2. Pre-calculate the Pseudo-Inverse
        T_pinv = torch.pinverse(T)
        
        self.register_buffer('T', T)
        self.register_buffer('T_pinv', T_pinv)

    def forward(self, x):
        B, L, C = x.shape
        x_trans = x.permute(0, 2, 1) 
        weights = torch.matmul(x_trans, self.T_pinv.transpose(0, 1))
        trend = torch.matmul(weights, self.T.transpose(0, 1))
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend
        

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

class UniversalMixer(nn.Module):
    def __init__(self, input_len, num_branches, hidden_dim=64):
        super().__init__()
        self.num_branches = num_branches
        self.input_len = input_len

        self.gating_network = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_branches)
        )

        self.cond_weight = nn.Parameter(torch.ones(input_len))
        self.cond_bias = nn.Parameter(torch.zeros(input_len))

    def forward(self, x_condition, outputs_list, pri=False):
        if len(outputs_list) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} outputs, got {len(outputs_list)}"
            )

        gate_logits = self.gating_network(x_condition)
        weights = torch.softmax(gate_logits, dim=-1)

        final_output = 0.0
        for i in range(self.num_branches):
            branch_out = outputs_list[i]
            w = weights[:, :, i].unsqueeze(-1)
            final_output = final_output + branch_out * w

            if pri:
                print(f"Branch {i} mean:", branch_out.mean(dim=(1, 2)))

        return final_output

# -----------------------------------------------------------------------------
# 2. Logic Blocks (Trend & Seasonality)
# -----------------------------------------------------------------------------

class TrendBlock(nn.Module):
    def __init__(self, configs, patch_len, stride, padding_layer):
        super().__init__()
        self.configs = configs
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = padding_layer
        
        # Dimensions
        patch_num_seq = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf_seq = configs.d_model * patch_num_seq
        
        # 1. Trend: Sequence Time
        self.trend_seq_time_encoder = build_tsti_encoder(configs, patch_num_seq, patch_len)
        self.trend_seq_time_head = FlattenHead(configs.enc_in, self.head_nf_seq, configs.pred_len, head_dropout=0)

        # 2. Trend: Vision Time
        self.trend_vit_time = timm.models.VisionTransformer(
            img_size=(patch_num_seq, patch_len),
            patch_size=(stride, stride),
            in_chans=1, 
            num_classes=0,
            embed_dim=configs.d_model,
            depth=3,
            num_heads=configs.n_heads,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        self.trend_vit_time_forecast = nn.Linear(configs.d_model, configs.pred_len)

        # Mixer
        self.trend_combiner = UniversalMixer(input_len=configs.seq_len, num_branches=2)

    def forward(self, x_trend):
        B, C, L = x_trend.shape

        # Pre-processing
        x_trend_patch = self.padding_patch_layer(x_trend)
        x_trend_patch = x_trend_patch.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Branch 1
        inp_trend_seq = x_trend_patch.permute(0, 1, 3, 2)
        enc_out_trend_seq = self.trend_seq_time_encoder(inp_trend_seq)
        out_trend_seq = self.trend_seq_time_head(enc_out_trend_seq)

        # Branch 2
        inp_trend_vis = torch.reshape(x_trend_patch, (B * C, x_trend_patch.shape[2], x_trend_patch.shape[3]))
        inp_trend_vis = inp_trend_vis.unsqueeze(1)
        
        x_enc_trend_vis = self.trend_vit_time.forward_features(inp_trend_vis)
        cls_trend_vis = x_enc_trend_vis[:, 0]
        y_trend_vis = self.trend_vit_time_forecast(cls_trend_vis)
        out_trend_vis = torch.reshape(y_trend_vis, (B, C, self.configs.pred_len))

        # Combine
        pred_trend = self.trend_combiner(x_trend, [out_trend_seq, out_trend_vis])
        
        return pred_trend


class SeasonalityBlock(nn.Module):
    def __init__(self, configs, patch_len, stride, padding_layer):
        super().__init__()
        self.configs = configs
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = padding_layer

        # Dimensions
        patch_num_seq = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf_seq = configs.d_model * patch_num_seq
        
        self.freq_len = configs.seq_len // 2 + 1
        patch_len_f = 16
        patch_num_freq = int((self.freq_len - patch_len_f) / stride + 2)
        head_nf_freq = configs.d_model * patch_num_freq

        # 1. Seas: Sequence Time
        self.seas_seq_time_encoder = build_tsti_encoder(configs, patch_num_seq, patch_len)
        self.seas_seq_time_head = FlattenHead(configs.enc_in, self.head_nf_seq, configs.pred_len, head_dropout=0)

        # 2. Seas: Sequence Frequency
        self.seas_seq_freq_encoder = build_tsti_encoder(configs, patch_num_freq, patch_len_f)
        self.seas_seq_freq_head = FlattenHead(configs.enc_in, head_nf_freq, configs.pred_len, head_dropout=0)

        # 3. Seas: Vision Time
        self.seas_vit_time = timm.models.VisionTransformer(
            img_size=(patch_num_seq, patch_len),
            patch_size=(stride, stride),
            in_chans=1, 
            num_classes=0,
            embed_dim=configs.d_model,
            depth=3,
            num_heads=configs.n_heads,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        self.seas_vit_time_forecast = nn.Linear(configs.d_model, configs.pred_len)

        # 4. Seas: Vision Frequency
        self.seas_vit_freq = timm.models.VisionTransformer(
            img_size=(patch_num_freq, patch_len_f),
            patch_size=(stride, stride),
            in_chans=2,
            num_classes=0,
            embed_dim=configs.d_model,
            depth=3,
            num_heads=configs.n_heads,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        self.seas_vit_freq_forecast = nn.Linear(configs.d_model, configs.pred_len * 2) 

        # Mixer
        self.seas_combiner = UniversalMixer(input_len=configs.seq_len, num_branches=4)

    def forward(self, x_seas):
        B, C, L = x_seas.shape

        # --- Pre-processing ---
        x_seas_patch = self.padding_patch_layer(x_seas)
        x_seas_patch = x_seas_patch.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        x_ft = torch.fft.rfft(x_seas, dim=-1)
        real = x_ft.real
        imag = x_ft.imag
        x_ft_stack = torch.stack((real, imag), dim=1) 
        x_ft_flat = x_ft_stack.reshape(B, 2 * C, -1)  

        x_freq_patch = self.padding_patch_layer(x_ft_flat)
        x_freq_patch = x_freq_patch.unfold(dimension=-1, size=16, step=self.stride)

        # Branches 1-4
        inp_seas_seq = x_seas_patch.permute(0, 1, 3, 2)
        enc_out_seas_seq = self.seas_seq_time_encoder(inp_seas_seq)
        out_seas_seq_t = self.seas_seq_time_head(enc_out_seas_seq)

        inp_seas_freq = x_freq_patch.permute(0, 1, 3, 2)
        enc_out_seas_freq = self.seas_seq_freq_encoder(inp_seas_freq)
        
        enc_out_p = torch.reshape(enc_out_seas_freq, (-1, C, enc_out_seas_freq.shape[-2], enc_out_seas_freq.shape[-1]))
        enc_out_p = enc_out_p.permute(0, 1, 3, 2)
        enc_out_p = self.seas_seq_freq_head(enc_out_p)

        enc_out_p = enc_out_p.reshape(B, 2, C, self.configs.pred_len)
        complex_out = torch.complex(enc_out_p[:, 0, :, :], enc_out_p[:, 1, :, :])
        out_seas_seq_f = torch.fft.irfft(complex_out, n=self.configs.pred_len)

        inp_seas_vis_t = torch.reshape(x_seas_patch, (B * C, x_seas_patch.shape[2], x_seas_patch.shape[3]))
        inp_seas_vis_t = inp_seas_vis_t.unsqueeze(1)
        
        x_enc_seas_vis_t = self.seas_vit_time.forward_features(inp_seas_vis_t)
        cls_seas_vis_t = x_enc_seas_vis_t[:, 0]
        y_seas_vis_t = self.seas_vit_time_forecast(cls_seas_vis_t)
        out_seas_vis_t = torch.reshape(y_seas_vis_t, (B, C, self.configs.pred_len))

        inp_seas_vis_f = x_freq_patch.view(B, 2, C, x_freq_patch.shape[2], x_freq_patch.shape[3])
        inp_seas_vis_f = inp_seas_vis_f.permute(0, 2, 1, 3, 4).contiguous() 
        inp_seas_vis_f = inp_seas_vis_f.view(B * C, 2, x_freq_patch.shape[2], x_freq_patch.shape[3])
        
        x_enc_seas_vis_f = self.seas_vit_freq.forward_features(inp_seas_vis_f)
        cls_seas_vis_f = x_enc_seas_vis_f[:, 0]
        y_seas_vis_f = self.seas_vit_freq_forecast(cls_seas_vis_f)
        
        y_seas_vis_f = torch.reshape(y_seas_vis_f, (B, C, 2, self.configs.pred_len))
        complex_vis_f = torch.complex(y_seas_vis_f[:, :, 0, :], y_seas_vis_f[:, :, 1, :])
        out_seas_vis_f = torch.fft.irfft(complex_vis_f, n=self.configs.pred_len)

        pred_seas = self.seas_combiner(x_seas, [out_seas_seq_t, out_seas_seq_f, out_seas_vis_t, out_seas_vis_f])
        
        return pred_seas

# -----------------------------------------------------------------------------
# 3. Main Combined Model
# -----------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.channels = configs.enc_in
        
        # Decomposition Block (Multi-Scale)
        self.decomposition = SeriesDecompPolynomial(self.seq_len)
        # self.decomposition = series_decomp_multi([7, 12, 14 ,24, 48])

        # Shared Padding Layer (for both seq and freq branches)
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        
        # Modular Blocks
        self.trend_block = TrendBlock(configs, patch_len, stride, self.padding_patch_layer)
        
        self.seas_block = SeasonalityBlock(configs, patch_len, stride, self.padding_patch_layer)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, C]

        # ---------------------------------------------
        # 1. RevIN (Normalization) on Input
        # ---------------------------------------------
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # ---------------------------------------------
        # 2. Decomposition (Multi-Scale) on Normalized Input
        # ---------------------------------------------
        x_seasonal_init, x_trend_init = self.decomposition(x_enc)

        # Prepare for branches: [B, C, L]
        # decomposition returns [B, L, C], blocks expect [B, C, L]
        x_trend = x_trend_init.permute(0, 2, 1)
        x_seas = x_seasonal_init.permute(0, 2, 1)

        # ---------------------------------------------
        # 3. Forward Pass through Modules
        # ---------------------------------------------
        pred_trend = self.trend_block(x_trend) # -> [B, C, PredLen]
        pred_seas = self.seas_block(x_seas)    # -> [B, C, PredLen]

        # ---------------------------------------------
        # 4. Denormalize & Combine
        # ---------------------------------------------
        # Sum components in normalized space
        dec_out = pred_trend + pred_seas # [B, C, PredLen]

        # Permute back to [B, PredLen, C] for denormalization
        dec_out = dec_out.permute(0, 2, 1)

        # Apply inverse normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None