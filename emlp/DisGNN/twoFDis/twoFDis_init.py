import torch.nn as nn
import torch
from ..basic_layers import Residual, Dense


class TwoFDisInit(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,   
                 k_tuple_dim: int,
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs
                 ):
        super().__init__()


        self.pattern_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=k_tuple_dim,
            padding_idx=0
        )
        
        self.mix_lin = Residual(
            hidden_dim=k_tuple_dim,
            activation_fn=activation_fn,
            mlp_num=2
        )
        


    def forward(self,
                ef: torch.Tensor,
                ):
        

        B = ef.shape[0]
        N = ef.shape[1]
        
        idx = torch.arange(N)
        tuple_pattern = torch.ones(size=(B, N, N), dtype=torch.int64, device=ef.device)
        tuple_pattern[:, idx, idx] = 2
        tuple_pattern = self.pattern_embedding(tuple_pattern) # (B, N, N, k_tuple_dim)
        
        emb2 = ef * tuple_pattern 
        
        emb2 = self.mix_lin(emb2)
        
        return emb2