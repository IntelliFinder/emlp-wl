#import pytorch_lightning as pl
import torch.nn as nn
from .Mol2Graph import Mol2Graph
from .utils.loss_fns import loss_fn_map
from .utils.activation_fns import activation_fn_map
from .utils.EMA import ExponentialMovingAverage
from .torch.nn.utils.clip_grad import clip_grad_norm_
import torch
from .twoFDis.twoFDis_init import TwoFDisInit
from .twoFDis.twoFDis_interaction import TwoFDisLayer
from .twoFDis.twoFDis_output import TwoOrderOutputBlock, TwoOrderDipOutputBlock, TwoOrderElcOutputBlock

from .basic_layers import Residual, Dense
from .utils.GradualWarmupScheduler import GradualWarmupScheduler


init_layer_dict = {
    "2FDis": TwoFDisInit,

}

interaction_layer_dict = {
    "2FDis": TwoFDisLayer,

}

output_layer_dict = {
    "2FDis": TwoOrderOutputBlock,
    "2FDisDip": TwoOrderDipOutputBlock,
    "2FDisElc": TwoOrderElcOutputBlock,

}

class kDisGNN(nn.Module):
    def __init__(self, 
        z_hidden_dim,
        ef_dim,
        rbf,
        max_z,
        rbound_upper,
        rbf_trainable,
        activation_fn,
        k_tuple_dim,
        block_num,
        pooling_level,
        e_mode,
        qm9,
        model_name,
        data_name,
        use_mult_lin,
        interaction_residual,
        global_y_mean,
        global_y_std,
    ):
        super().__init__()
        
        self.global_y_mean = global_y_mean
        self.global_y_std = global_y_std
        if qm9 and model_name == "2FDis" and (int(data_name) == 0):
            output_layer = output_layer_dict["2FDisDip"]
            self.global_y_std = 1.
            self.global_y_mean = 0.
            print("Using 2FDisDip as output layer")
        elif qm9 and model_name == "2FDis" and (int(data_name) == 5):
            output_layer = output_layer_dict["2FDisElc"]
            self.global_y_std = 1.
            self.global_y_mean = 0.
            print("Using 2FDisElc as output layer")
        else:
            output_layer = output_layer_dict[model_name]
        
        # Transform Molecule to Graph
        self.M2G = Mol2Graph(
            z_hidden_dim=z_hidden_dim,
            ef_dim=ef_dim,
            rbf=rbf,
            max_z=max_z,
            rbound_upper=rbound_upper,
            rbf_trainable=rbf_trainable
        )
        

        
        # initialize tuples
        init_layer = init_layer_dict[model_name]
        self.init_layer = init_layer(
            z_hidden_dim=z_hidden_dim,
            ef_dim=ef_dim,
            k_tuple_dim=k_tuple_dim,
            activation_fn=activation_fn,
        )
        
        # interaction layers
        self.interaction_layers = nn.ModuleList()
        if interaction_residual:
            self.interaction_residual_layers = nn.ModuleList()
        interaction_layer = interaction_layer_dict[model_name]
        for _ in range(block_num):
            self.interaction_layers.append(
                    interaction_layer(
                        hidden_dim=k_tuple_dim,
                        activation_fn=activation_fn,
                        e_mode=e_mode,
                        ef_dim=ef_dim,
                        use_mult_lin=use_mult_lin,
                        )
                    )
            if interaction_residual:
                self.interaction_residual_layers.append(
                    Residual(
                        mlp_num=2,
                        hidden_dim=k_tuple_dim,
                        activation_fn=activation_fn
                    )
                )

        # output layers
        self.output_layers = output_layer(
            hidden_dim=k_tuple_dim,
            activation_fn=activation_fn,
            pooling_level=pooling_level
            )
        
        
        self.predict_force = not qm9
        self.interaction_residual = interaction_residual


    def forward(self, dist_mat):
        #if self.predict_force: #TODO required in Hamiltonian?
        #    batch_data.pos.requires_grad_(True)
        
        # Molecule to Graphs
        ef = self.M2G(dist_mat)  
        kemb = self.init_layer(ef)

        # interaction
        for i in range(len(self.interaction_layers)):
            kemb = self.interaction_layers[i](
                kemb=kemb.clone(),
                ef=ef
                ) + kemb
            if self.interaction_residual:
                kemb = self.interaction_residual_layers[i](kemb)

        # output
        scores = self.output_layers(
            kemb=kemb,
            )
        
              
        return scores