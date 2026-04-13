import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.model import load_download_clip, Transformer


class MLPLayer(nn.Module):
    """
    LND - LND or ND - ND
    """

    def __init__(self, dim_list, dropout=0., activation='relu'):
        super().__init__()

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlp = nn.Sequential()

        for i in range(len(dim_list) - 2):
            _in = dim_list[i]
            _out = dim_list[i + 1]

            self.mlp.add_module(f"linear_{i}", nn.Linear(_in, _out))
            self.mlp.add_module(f"activate_{i}", self.activation_layer)
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(p=dropout))

        self.mlp.add_module(f"linear_final", nn.Linear(dim_list[-2], dim_list[-1]))

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPs(nn.Module):
    """
    Residual MLPs
    ***D - ***D
    """

    def __init__(self, org_dim, hidden_dim, dropout=0., num_layers=2, activation='relu'):
        super().__init__()
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(org_dim, hidden_dim),
            self.activation_layer,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, org_dim),
        ) for i in range(num_layers))

        self.lns = nn.ModuleList(nn.LayerNorm(org_dim) for i in range(num_layers))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x + self.mlps[i](self.lns[i](x))
        return x


class HashingEncoder(nn.Module):
    """
    hashing encoder, linear projection & tach.
    """

    def __init__(self, org_dim, k_bits, ):
        super().__init__()
        self.fc = nn.Linear(org_dim, k_bits)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class HashingDecoder(nn.Module):
    """
    hashing decoder, MLP & tach.
    """

    def __init__(self, org_bit_dim, recon_bit_dim):
        super().__init__()
        self.mlp = MLPLayer(dim_list=[org_bit_dim, recon_bit_dim, recon_bit_dim])

    def forward(self, x):
        return torch.tanh(self.mlp(x))


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class HashingModel(nn.Module):
    """
    Hashing model
    """

    def __init__(self, clip_info=None, args=None):
        super().__init__()

        self.dropout = dropout = args.dropout
        self.activation = activation = args.activation
        self.res_mlp_layers = res_mlp_layers = args.res_mlp_layers
        self.bit = bit = args.bit
        self.guide_bit_dim = guide_bit_dim = args.guide_bit_dim

        clip_embed_dim = clip_info['embed_dim']

        # predict layer
        self.img_cross_feat = nn.Sequential(
            nn.Linear(clip_embed_dim, clip_embed_dim * 4),
            QuickGELU(),
            # nn.GELU(),
            nn.Linear(clip_embed_dim * 4, clip_embed_dim),
        )
        self.txt_cross_feat = nn.Sequential(
            nn.Linear(clip_embed_dim, clip_embed_dim * 4),
            QuickGELU(),
            # nn.GELU(),
            nn.Linear(clip_embed_dim * 4, clip_embed_dim),
        )

        # share weight.
        self.resmlp_i = self.resmlp_t = ResidualMLPs(org_dim=clip_embed_dim, hidden_dim=4 * clip_embed_dim, dropout=dropout, num_layers=res_mlp_layers, activation=activation)

        # hash layer
        self.hash_encoder = HashingEncoder(org_dim=clip_embed_dim, k_bits=bit)

        # guide code
        self.guide_encoder = HashingEncoder(org_dim=clip_embed_dim, k_bits=guide_bit_dim)
        self.hash_decoder = HashingDecoder(bit, guide_bit_dim)

    def forward(self, img_cls, txt_eos, m1, m2):
        output_dict = {}

        output_dict['ori_img_feat'] = img_cls
        output_dict['ori_txt_feat'] = txt_eos
        pre_t_feat = self.img_cross_feat(img_cls)
        pre_i_feat = self.txt_cross_feat(txt_eos)

        output_dict['pre_i_feat'] = pre_i_feat
        output_dict['pre_t_feat'] = pre_t_feat
        mask1 = m1.expand_as(img_cls)
        mask2 = m2.expand_as(txt_eos)
        complete_i_feat = torch.where(mask1 == 0, pre_i_feat, img_cls)
        complete_t_feat = torch.where(mask2 == 0, pre_t_feat, txt_eos)

        res_img_cls = self.resmlp_i(complete_i_feat)
        res_txt_cls = self.resmlp_t(complete_t_feat)
        output_dict['after_res_img_cls'] = F.normalize(res_img_cls, dim=-1)
        output_dict['after_res_txt_cls'] = F.normalize(res_txt_cls, dim=-1)

        # hash code learn
        img_cls_hash = self.hash_encoder(res_img_cls)
        txt_cls_hash = self.hash_encoder(res_txt_cls)
        output_dict['img_cls_hash'] = img_cls_hash
        output_dict['txt_cls_hash'] = txt_cls_hash

        # guide hash code learn
        img_cls_guide = self.guide_encoder(res_img_cls)
        txt_cls_guide = self.guide_encoder(res_txt_cls)
        output_dict['img_cls_guide'] = img_cls_guide
        output_dict['txt_cls_guide'] = txt_cls_guide

        img_cls_hash_recon = self.hash_decoder(img_cls_hash)
        txt_cls_hash_recon = self.hash_decoder(txt_cls_hash)
        output_dict['img_cls_hash_recon'] = img_cls_hash_recon
        output_dict['txt_cls_hash_recon'] = txt_cls_hash_recon

        return output_dict


class PCEH(nn.Module):
    def __init__(self, args=None):
        super(PCEH, self).__init__()
        self.args = args
        self.clip, clip_info = load_download_clip(self.args.clip_path)

        # freeze CLIP
        if not self.args.is_train:
            for n, p in self.clip.named_parameters():
                p.requires_grad = False

        self.hash = HashingModel(clip_info=clip_info, args=args)

    def forward(self, image, text, key_padding_mask, m1, m2):
        _, _, img_cls = self.clip.encode_image(image)
        _, _, _, txt_eos = self.clip.encode_text(text, key_padding_mask)
        output_dict = self.hash(img_cls, txt_eos, m1, m2)
        return output_dict
