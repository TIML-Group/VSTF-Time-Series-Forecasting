import torch
from torch import nn
import torch.fft
import timm

# Imports as per the original files
from layers.Embed import PatchEmbedding
from layers.TSI_Encoder import *

# -----------------------------------------------------------------------------
# 1. Helper Modules (Shared)
# -----------------------------------------------------------------------------

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

        # Gating network
        self.gating_network = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_branches)
        )

        # Learnable affine params for condition normalization
        self.cond_weight = nn.Parameter(torch.ones(input_len))
        self.cond_bias = nn.Parameter(torch.zeros(input_len))

    def forward(self, x_condition, outputs_list, pri=False):
        """
        x_condition:  [B, C, L_in]
        outputs_list: List of N tensors, each [B, C, L_out]
        """
        if len(outputs_list) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} outputs, got {len(outputs_list)}"
            )

        # ---- 1. Compute gate weights ----
        gate_logits = self.gating_network(x_condition)
        weights = torch.softmax(gate_logits, dim=-1)

        # ---- 2. Normalize each branch output + weighted sum ----
        final_output = 0.0
        for i in range(self.num_branches):
            branch_out = outputs_list[i]
            w = weights[:, :, i].unsqueeze(-1)
            final_output = final_output + branch_out * w

            if pri:
                print(f"Branch {i} mean:", branch_out.mean(dim=(1, 2)))

        return final_output

# -----------------------------------------------------------------------------
# 2. Main Combined Model
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
        
        # Shared Padding Layers
        self.padding_patch_layer_seq = nn.ReplicationPad1d((0, stride))
        self.padding_patch_layer_freq = nn.ReplicationPad1d((0, stride))
        
        # =====================================================================
        # 1. Sequence Time Branch
        # =====================================================================
        patch_num_seq = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf_seq = configs.d_model * patch_num_seq
        
        self.seq_time_encoder = self._build_encoder(configs, patch_num_seq, patch_len)
        self.seq_time_head = FlattenHead(configs.enc_in, self.head_nf_seq, configs.pred_len, head_dropout=0)

        # =====================================================================
        # 2. Sequence Frequency Branch
        # =====================================================================
        self.freq_len = configs.seq_len // 2 + 1
        patch_len_f = 16
        patch_num_freq = int((self.freq_len - patch_len_f) / stride + 2)
        head_nf_freq = configs.d_model * patch_num_freq

        self.seq_freq_encoder = self._build_encoder(configs, patch_num_freq, patch_len_f)
        self.seq_freq_head = FlattenHead(configs.enc_in, head_nf_freq, configs.pred_len, head_dropout=0)

        # =====================================================================
        # 3. Vision Time Branch
        # =====================================================================
        self.vit_time = timm.models.VisionTransformer(
            img_size=(patch_num_seq, patch_len),
            patch_size=(stride, stride),
            in_chans=1, # Single channel for Time domain
            num_classes=0,
            embed_dim=configs.d_model,
            depth=3,
            num_heads=configs.n_heads,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        self.vit_time_forecast = nn.Linear(configs.d_model, configs.pred_len)

        # =====================================================================
        # 4. Vision Frequency Branch
        # =====================================================================
        # Applies Vision logic to Frequency Domain (Inputs: Real+Imag as 2 channels)
        self.vit_freq = timm.models.VisionTransformer(
            img_size=(patch_num_freq, patch_len_f),
            patch_size=(stride, stride),
            in_chans=2, # Two channels: Real and Imaginary
            num_classes=0,
            embed_dim=configs.d_model,
            depth=3,
            num_heads=configs.n_heads,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        # Output needs to predict complex values (Real+Imag), so *2
        self.vit_freq_forecast = nn.Linear(configs.d_model, configs.pred_len * 2) 

        # =====================================================================
        # Fusion
        # =====================================================================
        self.combiner = UniversalMixer(input_len=configs.seq_len, num_branches=4)


    def _build_encoder(self, configs, patch_num, patch_len):
        encoder = TSTiEncoder(configs.enc_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=2048,
                                n_layers=3, d_model=configs.d_model, n_heads=configs.n_heads, d_k=None, d_v=None, d_ff=configs.d_ff,
                                attn_dropout=0, dropout=configs.dropout, act="gelu", key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                   pe='zeros', learn_pe=True, verbose=False)
        return encoder

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # [B, L, C] -> [B, C, L]
        x_perm = x_enc.permute(0, 2, 1)
        B, C, L = x_perm.shape

        # -------------------------------------------------------
        # Common Pre-processing
        # -------------------------------------------------------
        # Time Domain Padding/Unfolding
        x_time_patch = self.padding_patch_layer_seq(x_perm)
        x_time_patch = x_time_patch.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x_time_patch: [B, C, PatchNum, PatchLen]

        # Frequency Domain FFT & Padding/Unfolding
        x_ft = torch.fft.rfft(x_perm, dim=-1)
        real = x_ft.real
        imag = x_ft.imag
        x_ft_stack = torch.stack((real, imag), dim=1) # [B, 2, C, F]
        x_ft_flat = x_ft_stack.reshape(B, 2 * C, -1)  # [B, 2*C, F]

        x_freq_patch = self.padding_patch_layer_freq(x_ft_flat)
        x_freq_patch = x_freq_patch.unfold(dimension=-1, size=16, step=self.stride) # patch_len_f=16
        # x_freq_patch: [B, 2*C, PatchNumF, PatchLenF]

        # -------------------------------------------------------
        # Branch 1: Sequence Time (Trend)
        # -------------------------------------------------------
        x_seq_t = x_time_patch.permute(0, 1, 3, 2) # [B, C, PatchLen, PatchNum] ? No, TSTi expects [B, C, N, P] -> permute inside usually?
        # Checking Temp_Freq logic: x.permute(0,1,3,2)
        x_seq_t = x_time_patch.permute(0, 1, 3, 2)
        
        enc_out_seq_t = self.seq_time_encoder(x_seq_t)
        out_seq_time = self.seq_time_head(enc_out_seq_t)

        # -------------------------------------------------------
        # Branch 2: Sequence Frequency
        # -------------------------------------------------------
        x_seq_f = x_freq_patch.permute(0, 1, 3, 2)
        enc_out_seq_f = self.seq_freq_encoder(x_seq_f)

        # Reshape for Head (handling the stacked channels)
        enc_out_p = torch.reshape(enc_out_seq_f, (-1, C, enc_out_seq_f.shape[-2], enc_out_seq_f.shape[-1]))
        enc_out_p = enc_out_p.permute(0, 1, 3, 2)
        enc_out_p = self.seq_freq_head(enc_out_p)

        # iFFT reconstruction
        enc_out_p = enc_out_p.reshape(B, 2, C, self.pred_len)
        complex_out = torch.complex(enc_out_p[:, 0, :, :], enc_out_p[:, 1, :, :])
        out_seq_freq = torch.fft.irfft(complex_out, n=self.pred_len)

        # -------------------------------------------------------
        # Branch 3: Vision Time
        # -------------------------------------------------------
        # Reshape: [B*C, 1, PatchNum, PatchLen]
        x_vis_t = torch.reshape(x_time_patch, (B * C, x_time_patch.shape[2], x_time_patch.shape[3]))
        x_vis_t = x_vis_t.unsqueeze(1) 
        
        x_enc_vis_t = self.vit_time.forward_features(x_vis_t)
        cls_vis_t = x_enc_vis_t[:, 0]
        y_vis_t = self.vit_time_forecast(cls_vis_t)
        out_vis_time = torch.reshape(y_vis_t, (B, C, self.pred_len))

        # -------------------------------------------------------
        # Branch 4: Vision Frequency
        # -------------------------------------------------------
        # Reshape: [B, 2*C, N, P] -> [B*C, 2, N, P]
        # We treat Real/Imag as the 2 channels for the Vision Transformer
        x_vis_f = x_freq_patch.view(B, 2, C, x_freq_patch.shape[2], x_freq_patch.shape[3])
        x_vis_f = x_vis_f.permute(0, 2, 1, 3, 4).contiguous() # [B, C, 2, N, P]
        x_vis_f = x_vis_f.view(B * C, 2, x_freq_patch.shape[2], x_freq_patch.shape[3])
        
        x_enc_vis_f = self.vit_freq.forward_features(x_vis_f)
        cls_vis_f = x_enc_vis_f[:, 0]
        y_vis_f = self.vit_freq_forecast(cls_vis_f) # [B*C, PredLen*2]
        
        # Reshape and iFFT
        y_vis_f = torch.reshape(y_vis_f, (B, C, 2, self.pred_len))
        complex_vis_f = torch.complex(y_vis_f[:, :, 0, :], y_vis_f[:, :, 1, :])
        out_vis_freq = torch.fft.irfft(complex_vis_f, n=self.pred_len)

        # -------------------------------------------------------
        # Combine & Denormalize
        # -------------------------------------------------------
        dec_out = self.combiner(x_perm, [out_seq_time, out_seq_freq, out_vis_time, out_vis_freq])
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None