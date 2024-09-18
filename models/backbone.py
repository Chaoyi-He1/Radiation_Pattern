from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from util.misc import NestedTensor

from .transformer import Transformer_Encoder, Transformer_Decoder, DropPath
from .positional_embedding import build_position_encoding


'''
Pointnet backbone based on Paper: 
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
https://arxiv.org/abs/1612.00593

Change the transform block to Transformer Encoder for unfixed point number,
The max number of points is 361.
The output of the backbone is the latent feature of the point cloud
The output latent feature is trained based on the contrastive loss
''' 
class Pointnet(nn.Module):
    def __init__(self, max_number_points: int = 361, 
                 input_dim: int = 3, 
                 output_dim: int = 1024, 
                 trans_num_layers: int = 4,   
                 drop_out_rate: float = 0.2, 
                 drop_path_rate: float = 0., 
                 norm_before: bool = True, 
                 activation: str = 'relu', 
                 weight_init: str = 'uniform', 
                 ) -> None:
        super().__init__()
        self.max_number_points = max_number_points
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_transform = Transformer_Encoder(
            num_layers=trans_num_layers,
            d_model=input_dim,
            nhead=1,
            dropout=drop_out_rate,
            drop_path=drop_path_rate,
            activation=activation,
            normalize_before=norm_before,
        ) 
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
        )
        self.pos_embed_feature = build_position_encoding(
            position_embedding_type='learned', 
            num_pos_feats=64,
        )
        self.feature_transform = Transformer_Encoder(
            num_layers=trans_num_layers,
            d_model=64,
            nhead=4,
            dropout=drop_out_rate,
            drop_path=drop_path_rate,
            activation=activation,
            normalize_before=norm_before,
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, output_dim),
        )
        self.sqz = nn.Conv1d(
            in_channels=max_number_points, 
            out_channels=1, 
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.weight_init(weight_init)
    
    def weight_init(self, mode='uniform'):
        '''
        Initialize the weights of the whole backbone
        '''
        if mode == 'uniform':
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        elif mode == 'normal':
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0, std=0.01)
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        The input of the backbone is the Tensor of point cloud 
            with data shape (B, N, 3)
        Let N is the number of points in each point cloud 
            where N <= 361 and N is different for each point cloud in B
        First, we put x into the full shape (B, 361, 3) with zero padding.
        Then, we generate the source mask for the input point cloud, 
            mask = 0 means the point is valid; mask = 1 means the point is invalid.
            We generate the mask based on the number of points in each point cloud.
        '''
        data = x
        output = self.input_transform(data)
        output = self.mlp1(output)
        pos_embed = self.pos_embed_feature(output)
        output = self.feature_transform(src=output, 
                                        pos=pos_embed)
        output = self.mlp2(output)
        output = self.sqz(output)
        return output.squeeze(1)
