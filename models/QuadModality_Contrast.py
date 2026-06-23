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

        self.gating_network = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_branches)
        )

        self.cond_weight = nn.Parameter(torch.ones(input_len))
        self.cond_bias = nn.Parameter(torch.zeros(input_len))

    def forward(self, x_condition, outputs_list, pri=False):
        # outputs_list contains tensors of shape [B, C, L]
        gate_logits = self.gating_network(x_condition)
        weights = torch.softmax(gate_logits, dim=-1) # [B, C, num_branches]

        # ⚡ OPTIMIZATION: Stack and use einsum instead of a python for-loop
        stacked_outputs = torch.stack(outputs_list, dim=-1) # [B, C, L, num_branches]
        final_output = torch.einsum('bcln,bcn->bcl', stacked_outputs, weights)

        if pri:
            for i in range(self.num_branches):
                print(f"Branch {i} mean:", outputs_list[i].mean(dim=(1, 2)))

        return final_output

# -----------------------------------------------------------------------------
# 2. Main Combined Model (Updated for SimCLR)
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

        self.d_model = configs.d_model
        self.n_heads = configs.n_heads

        self.o_d_model = 16
        self.o_n_heads = 4
        
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
        head_nf_freq = self.o_d_model * patch_num_freq

        configs.d_model = self.o_d_model
        configs.n_heads = self.o_n_heads
        configs.e_layers = 0

        self.seq_freq_encoder = self._build_encoder(configs, patch_num_freq, patch_len_f)
        self.seq_freq_head = FlattenHead(configs.enc_in, head_nf_freq, configs.pred_len, head_dropout=0)

        # =====================================================================
        # 3. Vision Time Branch
        # =====================================================================
        stride_3 = 8
        self.vit_time = timm.models.VisionTransformer(
            img_size=(patch_num_seq, patch_len),
            patch_size=(stride_3, stride_3),
            in_chans=1, # Single channel for Time domain
            num_classes=0,
            embed_dim=self.o_d_model,
            depth=0,
            num_heads=self.o_n_heads,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        self.vit_time_forecast = nn.Linear(self.o_d_model, self.pred_len)

        # =====================================================================
        # 4. Vision Frequency Branch
        # =====================================================================
        stride_4 = 8
        self.vit_freq = timm.models.VisionTransformer(
            img_size=(patch_num_freq, patch_len_f),
            patch_size=(stride_4, stride_4),
            in_chans=2, # Two channels: Real and Imaginary
            num_classes=0,
            embed_dim=self.o_d_model,
            depth=0,
            num_heads=self.o_n_heads,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        self.vit_freq_forecast = nn.Linear(self.o_d_model, configs.pred_len * 2) 

        # =====================================================================
        # Fusion
        # =====================================================================
        self.combiner = UniversalMixer(input_len=configs.seq_len, num_branches=4)

        # =====================================================================
        # SimCLR / Contrastive Projection Heads
        # =====================================================================
        # self.proj_dim = 128  # Shared latent space dimension, works for ETTh1 and ETTh2
        self.proj_dim = 128
        
        # Time Domain Heads (Mapping d_model -> proj_dim)
        self.st_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.proj_dim)
        )
        
        self.vt_proj = nn.Sequential(
            nn.Linear(self.o_d_model, self.o_d_model),
            nn.ReLU(),
            nn.Linear(self.o_d_model, self.proj_dim)
        )
        
        # Frequency Domain Heads
        # SF processes Real/Imag separately (2*C channels), flattening yields 2*d_model
        self.sf_proj = nn.Sequential(
            nn.Linear(self.o_d_model * 2, self.o_d_model), 
            nn.ReLU(),
            nn.Linear(self.o_d_model, self.proj_dim)
        )
        
        self.vf_proj = nn.Sequential(
            nn.Linear(self.o_d_model, self.o_d_model),
            nn.ReLU(),
            nn.Linear(self.o_d_model, self.proj_dim)
        )


    def _build_encoder(self, configs, patch_num, patch_len):
        encoder = TSTiEncoder(configs.enc_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=512,
                                n_layers=configs.e_layers, d_model=configs.d_model, n_heads=configs.n_heads, d_k=None, d_v=None, d_ff=configs.d_ff,
                                attn_dropout=0, dropout=configs.dropout, act="gelu", key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                   pe='zeros', learn_pe=True, verbose=False)
        return encoder

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_embeddings=False):
        # 1. Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_perm = x_enc.permute(0, 2, 1)
        B, C, L = x_perm.shape

        # -------------------------------------------------------
        # Common Pre-processing
        # -------------------------------------------------------
        x_time_patch = self.padding_patch_layer_seq(x_perm)
        x_time_patch = x_time_patch.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        x_ft = torch.fft.rfft(x_perm, dim=-1)
        
        # ⚡ OPTIMIZATION: view_as_real avoids separating and stacking real/imag explicitly
        x_ft_stack = torch.view_as_real(x_ft).permute(0, 3, 1, 2) # [B, 2, C, F]
        x_ft_flat = x_ft_stack.reshape(B, 2 * C, -1)  

        x_freq_patch = self.padding_patch_layer_freq(x_ft_flat)
        x_freq_patch = x_freq_patch.unfold(dimension=-1, size=16, step=self.stride) 

        # -------------------------------------------------------
        # Branch 1: Sequence Time (Trend)
        # -------------------------------------------------------
        x_seq_t = x_time_patch.permute(0, 1, 3, 2)
        enc_out_seq_t = self.seq_time_encoder(x_seq_t) 
        out_seq_time = self.seq_time_head(enc_out_seq_t)
        
        st_features = enc_out_seq_t.mean(dim=-1).view(B * C, self.d_model)

        # -------------------------------------------------------
        # Branch 2: Sequence Frequency
        # -------------------------------------------------------
        x_seq_f = x_freq_patch.permute(0, 1, 3, 2)
        enc_out_seq_f = self.seq_freq_encoder(x_seq_f) 

        sf_features = enc_out_seq_f.mean(dim=-1) 
        sf_features = sf_features.view(B, 2, C, self.o_d_model)
        sf_features = sf_features.permute(0, 2, 1, 3).contiguous().view(B * C, 2 * self.o_d_model)

        enc_out_p = torch.reshape(enc_out_seq_f, (-1, C, enc_out_seq_f.shape[-2], enc_out_seq_f.shape[-1]))
        enc_out_p = enc_out_p.permute(0, 1, 3, 2)
        enc_out_p = self.seq_freq_head(enc_out_p)

        # ⚡ OPTIMIZATION: view_as_complex directly interprets memory instead of allocating new tensor
        enc_out_p = enc_out_p.reshape(B, 2, C, self.pred_len)
        complex_out = torch.view_as_complex(enc_out_p.permute(0, 2, 3, 1).contiguous())
        out_seq_freq = torch.fft.irfft(complex_out, n=self.pred_len)

        # -------------------------------------------------------
        # Branch 3: Vision Time
        # -------------------------------------------------------
        # ⚡ OPTIMIZATION: Combine reshape and unsqueeze into a single reshape
        x_vis_t = x_time_patch.reshape(B * C, 1, x_time_patch.shape[2], x_time_patch.shape[3])
        x_enc_vis_t = self.vit_time.forward_features(x_vis_t)
        
        cls_vis_t = x_enc_vis_t[:, 0]
        vt_features = cls_vis_t
        
        y_vis_t = self.vit_time_forecast(cls_vis_t)
        out_vis_time = torch.reshape(y_vis_t, (B, C, self.pred_len))

        # -------------------------------------------------------
        # Branch 4: Vision Frequency
        # -------------------------------------------------------
        x_vis_f = x_freq_patch.view(B, 2, C, x_freq_patch.shape[2], x_freq_patch.shape[3])
        x_vis_f = x_vis_f.permute(0, 2, 1, 3, 4).contiguous() 
        x_vis_f = x_vis_f.view(B * C, 2, x_freq_patch.shape[2], x_freq_patch.shape[3])
        
        x_enc_vis_f = self.vit_freq.forward_features(x_vis_f)
        
        cls_vis_f = x_enc_vis_f[:, 0]
        vf_features = cls_vis_f
        
        y_vis_f = self.vit_freq_forecast(cls_vis_f) 
        
        # ⚡ OPTIMIZATION: view_as_complex
        y_vis_f = torch.reshape(y_vis_f, (B, C, 2, self.pred_len))
        complex_vis_f = torch.view_as_complex(y_vis_f.permute(0, 1, 3, 2).contiguous())
        out_vis_freq = torch.fft.irfft(complex_vis_f, n=self.pred_len)

        # -------------------------------------------------------
        # Combine & Denormalize
        # -------------------------------------------------------
        dec_out = self.combiner(x_perm, [out_seq_time, out_seq_freq, out_vis_time, out_vis_freq])
        dec_out = dec_out.permute(0, 2, 1)

        # ⚡ OPTIMIZATION: Rely on native broadcasting. No `.repeat()` required. 
        # Saves immense memory overhead.
        dec_out = dec_out * stdev + means

        # -------------------------------------------------------
        # Return Projected Embeddings for SimCLR
        # -------------------------------------------------------
        if return_embeddings:
            z_st = self.st_proj(st_features)
            z_vt = self.vt_proj(vt_features)
            
            z_sf = self.sf_proj(sf_features)
            z_vf = self.vf_proj(vf_features)
            
            return dec_out, (z_st, z_vt, z_sf, z_vf)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_embeddings=True):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if return_embeddings:
                dec_out, embeddings = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_embeddings=True)
                return dec_out[:, -self.pred_len:, :], embeddings
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
        return None