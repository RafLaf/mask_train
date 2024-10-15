import torch.nn as nn
import torch
from torch import Tensor
import numpy as np

#monkey patch of forward function in self.model.blocks[i].attn with 
'''
def custom_attn_forward(self, x: Tensor) -> Tensor:
    print(x.shape,'x')
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    print(self.qkv(x).shape)
    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1)

    attn = attn.softmax(dim=-1)
    
    #print(attn.mean(0),attn.mean(1))
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
'''





class DINO_V2(nn.Module):
    def __init__(self, name):
        super().__init__()
        try:
            self.model = torch.hub.load('/lustre/fswork/projects/rech/csb/ude77uf/hub/facebookresearch_dinov2_main', name, source='local')
        except:
            self.model = torch.hub.load('facebookresearch/dinov2', name)
        self.model = self.model.float()
        self.outdim = self.model.embed_dim
        self.num_heads = self.model.blocks[0].attn.num_heads
        self.head_dim = self.outdim//self.num_heads
        self.name = 'dinov2'
        ones = torch.ones(len(self.model.blocks), self.num_heads)*1.0
        self.pruning_mask = nn.Parameter(ones)
        self.scale = 0
        self.qkv = 0
        self.position = 0


    def apply_pruning_to_vit(self, prunning_matrix):

        for i in range(int(prunning_matrix.shape[0])):
            for j in range(int(prunning_matrix.shape[1])):
                if prunning_matrix[i,j] == 0:
                    idx = torch.zeros(self.outdim)
                    idx[j*self.head_dim:(j+1)*self.head_dim] = 1
                    idx = idx.bool()
                    self.model.blocks[i].attn.qkv.weight.data[:,idx] = 0
                    #QKV(3D,D) * X (D) --> Y (3D)
                    #to not put the ouput at 0 but simply remove the effect of one head I should prune in the second dimension.
                    #original_attn_layer = self.model.blocks[i].attn
                    #original_attn_layer.forward  = custom_attn_forward.__get__(original_attn_layer, nn.Module)
    def apply_pruning_old(self):
        """
        Apply the pruning mask in a differentiable way.
        Each head will have a scaling factor learned through the pruning mask.
        """
        num_layers = self.pruning_mask.shape[0]
        num_heads = self.pruning_mask.shape[1]
        
        # Iterate over each layer and each head
        for i in range(num_layers):
            for j in range(num_heads):
                # Get the mask value for this head
                idx = torch.zeros(self.outdim)
                idx[j * self.head_dim : (j + 1) * self.head_dim] = 1
                idx = idx.bool()
                #self.model.blocks[i].attn.qkv.weight.data[:, idx] *= torch.sigmoid(self.pruning_mask[i, j])
                self.model.blocks[i].attn.qkv.weight.data[:, idx] *= self.pruning_mask[i, j]
    
    def custom_attn_forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        
        q, k, v = q * self.pruning_mask[self.position].view(1,-1,1,1), k * self.pruning_mask[self.position].view(1,-1,1,1), v * self.pruning_mask[self.position].view(1,-1,1,1)
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def apply_pruning_mask(self):
        num_layers = self.pruning_mask.shape[0]
        num_heads = self.pruning_mask.shape[1]
        
        # Iterate over each layer and each head
        for i in range(num_layers):
            scale = self.model.blocks[i].attn.scale
            qkv = self.model.blocks[i].attn.qkv
            attn_drop = self.model.blocks[i].attn.attn_drop
            proj = self.model.blocks[i].attn.proj
            proj_drop = self.model.blocks[i].attn.proj_drop
            position = i
            
            # Define a custom forward function for this layer
            def custom_attn_forward(self, x: torch.Tensor, p_mask = self.pruning_mask[i].view(1,-1,1,1),scale=scale, qkv=qkv, attn_drop=attn_drop, proj=proj, proj_drop=proj_drop, position=position) -> torch.Tensor:
                B, N, C = x.shape
                qkv_out = qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv_out[0] * scale, qkv_out[1], qkv_out[2]
                #p_mask = torch.clip(p_mask, 0.0, 1.0)
                
                q, k, v = q * p_mask, k * p_mask, v * p_mask
                #p_mask_s = steep_sigmoid(p_mask)
                #q, k, v = q * p_mask_s, k * p_mask_s, v * p_mask_s
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = proj(x)
                x = proj_drop(x)
                return x

            # Assign the custom forward method to the current layer
            self.model.blocks[i].attn.forward = custom_attn_forward.__get__(self.model.blocks[i].attn, nn.Module)

    
    def forward(self, x):
        return self.model.forward(x)


def create_model(name='dinov2_vitb14'):
    return DINO_V2(name)


def steep_sigmoid(x, k=10):
    """
    Steeper sigmoid-like function that satisfies f(0) = 0 and f(1) = 1.
    :param x: Input tensor or value, expected to be between 0 and 1.
    :param k: The steepness parameter. The larger k, the steeper the curve.
    :return: Transformed value or tensor with the desired shape.
    """
    # Apply the scaled sigmoid function
    sigmoid_part = torch.sigmoid(k * (x - 0.5))
    
    # Rescale to ensure f(0) = 0 and f(1) = 1
    return (sigmoid_part - torch.sigmoid(torch.tensor(-k / 2))) / (1 - 2 * torch.sigmoid(torch.tensor(-k / 2)))