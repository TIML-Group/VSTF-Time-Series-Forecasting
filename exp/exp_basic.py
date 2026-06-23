import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD, PatchTST_Decomp, PatchTST_Raw, PatchTST_Noencoder, PatchTST_MoE, PatchTST_MoEPatch_Decomp, \
    PatchTST_Decomp_MoEPatch, ViT_MAEFrozen, VisionTS, Seq_Vision, TemporalFreq_trend2sea2, TemporalFreq_trend2sea4, TemporalFreq_trend2sea4_revinv, VSTF0, Temp_Freq, QuadModality, QuadModality_minus_ViT, PatchTST_RevIn, PatchTST_Attention, Temp_Freq_Concat, VSTF5_orginal, VSTF_Final


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
            "PatchTST_Decomp": PatchTST_Decomp,
            "PatchTST_Raw": PatchTST_Raw,
            "PatchTST_MoE": PatchTST_MoE,
            "PatchTST_Noencoder": PatchTST_Noencoder,
            "PatchTST_MoEPatch_Decomp": PatchTST_MoEPatch_Decomp,
            "PatchTST_Decomp_MoEPatch": PatchTST_Decomp_MoEPatch,
            "ViT_MAEFrozen": ViT_MAEFrozen,
            "VisionTS": VisionTS,
            "Seq_Vision": Seq_Vision,
            "TemporalFreq_trend2sea2": TemporalFreq_trend2sea2,
            "TemporalFreq_trend2sea4": TemporalFreq_trend2sea4,
            "TemporalFreq_trend2sea4_revinv": TemporalFreq_trend2sea4_revinv,
            "VSTF0": VSTF0,
            "Temp_Freq": Temp_Freq,
            "QuadModality": QuadModality,
            "QuadModality_minus_ViT": QuadModality_minus_ViT,
            "PatchTST_RevIn": PatchTST_RevIn,
            "PatchTST_Attention": PatchTST_Attention,
            "Temp_Freq_Concat": Temp_Freq_Concat,
            "VSTF5_orginal": VSTF5_orginal,
            "VSTF_Final": VSTF_Final,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
