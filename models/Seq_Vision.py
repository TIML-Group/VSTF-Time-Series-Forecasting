import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import timm
from layers.TSI_Encoder import *
import matplotlib.pyplot as plt



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

        # patching and embedding
        self.padding_patch_layer_seq = nn.ReplicationPad1d((0, stride))
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, self.padding, configs.dropout)
        patch_num = int((configs.seq_len - patch_len) / stride + 2)

        # Encoder
        self.trend_encoder = TSTiEncoder(configs.enc_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=2048,
                                n_layers=3, d_model=configs.d_model, n_heads=configs.n_heads, d_k=None, d_v=None, d_ff=configs.d_ff,
                                attn_dropout=0, dropout=configs.dropout, act="gelu", key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                   pe='zeros', learn_pe=True, verbose=False)
        

        # Imaging Components
        self.vit = timm.models.VisionTransformer(
            img_size=(patch_num, patch_len),
            patch_size=(stride, stride),
            in_chans=1,
            num_classes=0,  # No classification head
            embed_dim=configs.d_model,
            depth=3,
            num_heads=configs.n_heads,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        self.vit_forecast = nn.Linear(configs.d_model, configs.pred_len)

        self.combiner = UniversalMixer(input_len=configs.seq_len, num_branches=2)

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.trend_head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=0)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_perm = x_enc.permute(0, 2, 1)

        # Sequence
        x_enc_seq = self.padding_patch_layer_seq(x_perm)
        x_enc_seq = x_enc_seq.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_enc_seq = x_enc_seq.permute(0,1,3,2)
        enc_out = self.trend_encoder(x_enc_seq)
        num_out = self.trend_head(enc_out)

        # Imaging
        B, C, L = x_perm.shape
        x = self.padding_patch_layer(x_perm)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        plt.figure(figsize=(3, 6))
        plt.imshow(x[0, :, :, :].detach().cpu().squeeze().numpy(), cmap='gray', interpolation='nearest')
        # Optional: Cleanup for paper figures
        plt.axis('off')  # Turn off axis numbers
        plt.title('Grayscale Feature Map')
        plt.savefig("feature_map.png", bbox_inches='tight', pad_inches=0, dpi=300)
        sys.exit()
        x_img_enc = self.vit.forward_features(x)
        cls_token_embedding = x_img_enc[:, 0]

        y = self.vit_forecast(cls_token_embedding)
        image_out = torch.reshape(y, (B, C, self.pred_len))
        
        dec_out = self.combiner(x_perm, [num_out, image_out])
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
