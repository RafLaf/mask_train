import os
import sys

# Obtenez le chemin du dossier courant où se trouve ce script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin relatif vers le dossier CLIP dans votre projet
clip_path = os.path.join(current_dir, "../../../CLIP")

# Ajoutez ce chemin à sys.path
sys.path.insert(0, clip_path)

import clip
import torch.nn as nn

class Clip(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model, _ = clip.load(name)
        self.model = self.model.float()
        self.outdim = self.model.visual.output_dim
        self.name = 'clip'

    def forward(self, x):
        return self.model.encode_image(x)


def create_model(name="ViT-B/16"):
    return Clip(name)



    

