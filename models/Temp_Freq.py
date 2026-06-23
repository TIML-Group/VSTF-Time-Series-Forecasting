import torch
from torch import nn
import torch.fft
import timm
import gc

# Keep original imports
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, ComplexToRealEmbedding, RealToComplexEmbedding
from layers.yy import FourierBlock3
from layers.TSI_Encoder import *
import matplotlib.pyplot as plt

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


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class UniversalMixer(nn.Module):
    def __init__(self, input_len, num_branches, hidden_dim=64, bias_init=None, constrain_main_branch=True):
        """
        Args:
            input_len: Length of condition vector (C dimension).
            num_branches: Number of experts/branches.
            hidden_dim: Hidden dimension of gating MLP.
            prior_probs: List of floats summing to 1 (e.g., [0.7, 0.3]). 
                         Sets initial bias. If None, defaults to equal.
            constrain_main_branch: If True, forces Branch 0 probability to be [0.5, 1.0].
        """
        super().__init__()
        self.num_branches = num_branches
        self.constrain_main = constrain_main_branch
        
        # 1. Pre-norm: Vital for time series inputs
        self.input_norm = nn.LayerNorm(input_len)

        # 2. Gating Network with SiLU
        self.gating_network = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # Smooth activation
            nn.Linear(hidden_dim, num_branches)
        )

        if bias_init is not None:
                # bias_init should be a list, e.g., [5.0, -5.0] to favor branch 0
                assert len(bias_init) == num_branches
                # Access the last linear layer
                final_linear = self.gating_network[-1]
                with torch.no_grad():
                    final_linear.bias.copy_(torch.tensor(bias_init, dtype=torch.float))
                    final_linear.weight.data.fill_(0.01) # Keep weights small initially
        
    def forward(self, x_condition, outputs_list):
        """
        x_condition:  [B, C, L_in] (Gating input)
        outputs_list: List of N tensors, each [B, C, L_out]
        """
        if len(outputs_list) != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} branches, got {len(outputs_list)}")

        # 1. Normalize Condition
        # x_norm = self.input_norm(x_condition)
        x_norm = x_condition

        # 2. Compute Raw Probabilities
        gate_logits = self.gating_network(x_norm) 
        weights = torch.softmax(gate_logits, dim=-1) # Range [0, 1]

        # 4. Weighted Sum (Vectorized)
        # Stack: [B, C, L_out, N]
        stacked_outputs = torch.stack(outputs_list, dim=-1)
        
        # Expand weights: [B, C, N] -> [B, C, 1, N] for broadcasting
        weights_expanded = weights.unsqueeze(2)
        
        # Sum over branches
        final_output = torch.sum(stacked_outputs * weights_expanded, dim=-1)

        return final_output

# -----------------------------------------------------------------------------
# 2. Main Model
# -----------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        self.stride = stride

        self.padding_patch_layer_seq = nn.ReplicationPad1d((0, stride))
        self.padding_patch_layer_freq = nn.ReplicationPad1d((0, stride))

        # --------------------
        # Trend Branch Components
        # --------------------
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)

        patch_num_trend = int((configs.seq_len - patch_len) / stride + 2)
        self.trend_encoder = self._build_encoder(configs, patch_num_trend, patch_len)
        head_nf_trend = configs.d_model * patch_num_trend
        self.trend_head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=0)

        # --------------------
        # Seasonal/Main Branch Components
        # --------------------
        self.freq_len = configs.seq_len // 2 + 1
        patch_len_f = 16;
        stride_f = 8
        patch_num_freq = int((self.freq_len - patch_len) / stride + 2)
        head_f_trend = configs.d_model * patch_num_freq

        self.seasonal_freq_encoder = self._build_encoder(configs, patch_num_freq, patch_len_f)
        self.seasonal_freq_head = FlattenHead(configs.enc_in, head_f_trend, configs.pred_len,
                                    head_dropout=0)

        self.trend_combiner = UniversalMixer(input_len=self.seq_len, num_branches=2, bias_init=[3.0, -3.0])


    def _build_encoder(self, configs, patch_num, patch_len):
        encoder = TSTiEncoder(configs.enc_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=2048,
                                n_layers=3, d_model=configs.d_model, n_heads=configs.n_heads, d_k=None, d_v=None, d_ff=configs.d_ff,
                                attn_dropout=0, dropout=configs.dropout, act="gelu", key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                   pe='zeros', learn_pe=True, verbose=False)
        return encoder

        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. Manual Normalization (Original Input)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # [B, L, C] -> [B, C, L]
        x_perm = x_enc.permute(0, 2, 1)
        # 2. Path A: Transformer
        x_enc = self.padding_patch_layer_seq(x_perm)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_enc = x_enc.permute(0,1,3,2)
        enc_out = self.trend_encoder(x_enc)
        out_trans = self.trend_head(enc_out)

        # 2. FFT
        x_ft = torch.fft.rfft(x_perm, dim=-1)

        # --- Path A: Frequency Transformer ---
        real = x_ft.real;
        imag = x_ft.imag
        x_enc_c = torch.stack((real, imag), dim=1)
        B, Two, C, F = x_enc_c.shape
        x_enc_c = x_enc_c.reshape(B, Two * C, F)

        # Plot
        x_freq_patch = self.padding_patch_layer_freq(x_enc_c)
        x_freq_patch = x_freq_patch.unfold(dimension=-1, size=16, step=self.stride)
        x_vis_f = x_freq_patch.view(B, 2, C, x_freq_patch.shape[2], x_freq_patch.shape[3])
        x_vis_f = x_vis_f.permute(0, 2, 1, 3, 4).contiguous() # [B, C, 2, N, P]
        x_vis_f = x_vis_f.view(B * C, 2, x_freq_patch.shape[3], x_freq_patch.shape[2])
        data = x_vis_f[0, :, :, :].squeeze(0).detach().cpu().squeeze().numpy()
        height, width = data.shape[1], data.shape[2]
        rgb_image = np.zeros((height, width, 3))
        
        # 2. Normalize data to 0-1 range (Crucial for RGB plotting)
        # If your data is unbounded, normalize it relative to the max value in the array
        max_val = data.max()
        norm_data = data / max_val
        
        # 3. Assign Channels
        rgb_image[..., 0] = norm_data[0]  # Red Channel
        rgb_image[..., 1] = norm_data[1]  # Green Channel
        # rgb_image[..., 2] remains 0 (Blue)
        
        # 4. Plot
        plt.figure(figsize=(4, 2))
        plt.imshow(rgb_image, aspect='auto')
        plt.title("Red=Ch0, Green=Ch1, Yellow=Overlap")
        plt.axis('off')
        plt.savefig("spectrum.png", bbox_inches='tight', pad_inches=0, dpi=300)
        sys.exit()
        
        x_enc_freq = self.padding_patch_layer_freq(x_enc_c)
        x_enc_freq = x_enc_freq.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_enc_freq = x_enc_freq.permute(0,1,3,2)
        enc_out_freq = self.seasonal_freq_encoder(x_enc_freq)

        enc_out_p = torch.reshape(enc_out_freq, (-1, C, enc_out_freq.shape[-2], enc_out_freq.shape[-1]))
        enc_out_p = enc_out_p.permute(0, 1, 3, 2)
        enc_out_p = self.seasonal_freq_head(enc_out_p)

        enc_out_p = enc_out_p.reshape(B, 2, C, self.pred_len)
        enc_out_p = torch.complex(enc_out_p[:, 0, :, :], enc_out_p[:, 1, :, :])
        dec_out = torch.fft.irfft(enc_out_p, n=self.pred_len)

        # 4. Combine
        # dec_out = out_trans + self.trend_combiner(torch.cat((real, imag), dim=-1), [out_trans, dec_out])
        dec_out = out_trans + self.trend_combiner(x_perm, [out_trans, dec_out])
        dec_out = dec_out.permute(0, 2, 1)

        # 5. Manual Denormalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]