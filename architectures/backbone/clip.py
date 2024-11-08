import os
import sys
import torch

# Obtenez le chemin du dossier courant où se trouve ce script
#current_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin relatif vers le dossier CLIP dans votre projet
#clip_path = os.path.join(current_dir, "../../../CLIP")

# Ajoutez ce chemin à sys.path
#sys.path.insert(0, clip_path)

import clip
import torch.nn as nn
from architectures.clip_utils import MultiheadAttention


class Clip(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model, _ = clip.load(name)
        self.model = self.model.float()
        self.outdim = self.model.visual.output_dim
        self.name = 'clip'
        self.num_heads = self.model.visual.transformer.resblocks[0].attn.num_heads
        self.num_layers = len(self.model.visual.transformer.resblocks)
        self.d_model = self.model.visual.transformer.width
        zeros = torch.zeros(self.num_layers, self.num_heads).to(device='cuda')
        self.pruning_mask = nn.Parameter(zeros)

        reweight_heads = True
        if reweight_heads:
            for l in range(self.num_layers):
                self.model.visual.transformer.resblocks[l].attn = MultiheadAttention(self.d_model, self.num_heads)
                self.model.visual.transformer.resblocks[l].attn.mask_v = self.pruning_mask[l]
    

    def apply_pruning_mask(self):
        pass

    def forward(self, x):
        return self.model.encode_image(x)


def create_model(name="ViT-B/16"):
    return Clip(name)



    

