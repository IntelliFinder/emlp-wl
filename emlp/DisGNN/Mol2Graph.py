import torch.nn as nn
from torch import Tensor
import torch
from .basis_layers import rbf_class_mapping


class Mol2Graph(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 rbf: str,
                 rbf_trainable: bool,
                 rbound_upper: float,
                 max_z: int,
                 **kwargs):
        super().__init__()
        self.rbf_fn = rbf_class_mapping[rbf](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=rbf_trainable,
                    **kwargs
                )
        self.z_emb = nn.Embedding(max_z + 1, z_hidden_dim, padding_idx=0)
        

    def forward(self, dist_mat: Tensor, **kwargs):
        '''
            dist_mat (B, N, N, 1)
        '''
        
        B, N = dist_mat.shape[0], dist_mat.shape[1]
       
        ef = self.rbf_fn(dist_mat.reshape(-1, 1)).reshape(B, N, N, -1) # (B, N, N, ef_dim)
        
        return ef


