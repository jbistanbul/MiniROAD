import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .Transformer import TransformerModel
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
from model.model_builder import META_ARCHITECTURES as registry

FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048
}

@registry.register('Transformer')
class ViTEnc(nn.Module):
    def __init__(self, cfg, use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned"
    ):
        super(ViTEnc, self).__init__()
        self.img_dim = cfg["window_size"]
        self.out_dim = cfg["num_classes"]
        self.embedding_dim = cfg["embedding_dim"]
        self.patch_dim = cfg["patch_dim"]
        self.num_heads = cfg["num_heads"]
        self.num_layers = cfg["num_layers"]
        self.hidden_dim = cfg["hidden_dim"]
        self.dropout_rate = cfg["dropout"]
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        self.num_channels= 0
        if self.use_rgb:
            self.num_channels += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.num_channels += FEATURE_SIZES[cfg['flow_type']]
        self.attn_dropout_rate = cfg["attn_dropout_rate"]
        assert self.embedding_dim % self.num_heads == 0
        assert self.img_dim % self.patch_dim == 0
        self.conv_patch_representation = conv_patch_representation
        self.num_patches = int(self.img_dim // self.patch_dim)
        self.seq_length = self.num_patches + 1
        # self.seq_length = self.num_patches + self.img_dim
        self.flatten_dim = self.patch_dim * self.patch_dim * self.num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        # self.cls_token = nn.Parameter(torch.zeros(1, self.img_dim, self.embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position encoding :', positional_encoding_type)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TransformerModel(
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)

        use_representation = False  # False
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(self.embedding_dim , self.hidden_dim//2),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(self.hidden_dim//2, self.out_dim),
            )
        else:
            self.mlp_head = nn.Linear(self.embedding_dim, self.out_dim)

        if self.conv_patch_representation:
            # self.conv_x = nn.Conv2d(
            #     self.num_channels,
            #     self.embedding_dim,
            #     kernel_size=(self.patch_dim, self.patch_dim),
            #     stride=(self.patch_dim, self.patch_dim),
            #     padding=self._get_padding(
            #         'VALID', (self.patch_dim, self.patch_dim),
            #     ),
            # )
            self.conv_x = nn.Conv1d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                padding=self._get_padding(
                    'VALID',  (self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()


    def forward(self, sequence_input_rgb, sequence_input_flow):
        if self.use_rgb and self.use_flow:
            x = torch.cat((sequence_input_rgb, sequence_input_flow), 2)
        elif self.use_rgb:
            x = sequence_input_rgb
        elif self.use_flow:
            x = sequence_input_flow

        x = self.linear_encoding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # B, 1, 1024
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((x, cls_tokens), dim=1) # B, seq+1, 1024
        x = self.position_encoding(x) # B, seq+1, 1024
        x = self.pe_dropout(x)   # not delete

        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # B, seq+1, 1024

        x = self.to_cls_token(x[:, 0]) # B, 1024
        # x = self.to_cls_token(x[:,0:self.img_dim]) # B, 1024
        x = self.mlp_head(x)
        # x = F.log_softmax(x, dim=-1)
        out_dict = {}
        out_dict['logits'] = x.unsqueeze(1)
        # out_dict['logits'] = x #x.unsqueeze(1)
        return out_dict

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

