"""
module for gradient-based test-time methods, e.g., finetune, eTT, TSA, URL, Cosine Classifier
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
from peft import LoraConfig, get_peft_model
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import math
from torch import Tensor
from architectures.classifier.LR import Logistic_Regression




class FinetuneModel(torch.nn.Module):
    """
    the overall finetune module that incorporates a backbone and a head.
    """
    def __init__(self, backbone, way, device):
        super().__init__()
        '''
        backbone: the pre-trained backbone
        way: number of classes
        device: GPU ID
        use_alpha: for TSA only. Whether use adapter to adapt the backbone.
        use_beta: for URL and TSA only. Whether use  pre-classifier transformation.
        head: Use fc head or PN head to finetune.'''


        self.backbone = backbone

        self.L = nn.Linear(backbone.outdim, way).to(device)


    def forward(self, x, backbone_grad = True):
        # turn backbone_grad off if backbone is not to be finetuned
        self.backbone.apply_pruning_mask()
        if backbone_grad:
            x = self.backbone(x)
        else:
            with torch.no_grad():
                x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.L(x)
        return x

class Mask(torch.nn.Module):
    """
    the overall finetune module that incorporates a backbone and a head.
    """
    def __init__(self,config):
        super().__init__()
        '''
        backbone: the pre-trained backbone
        way: number of classes
        device: GPU ID
        use_alpha: for TSA only. Whether use adapter to adapt the backbone.
        use_beta: for URL and TSA only. Whether use  pre-classifier transformation.
        head: Use fc head or PN head to finetune.
        '''
        self.model = None
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)

        
        self.ft_epoch2 = config.MODEL.EPOCH_MASK
        self.lr = config.MODEL.LR_MASK
        self.batch_size = 64


    def initialize_model(self, way):
        '''
        Initializes the FinetuneModel if not already initialized or if the number of classes (way) changes.
        '''
        if self.model is None or self.model.L.out_features != way:
            self.model = FinetuneModel(self.backbone, way, device='cuda')

        self.L = nn.Linear(self.backbone.outdim, way)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.backbone.pruning_mask.requires_grad = True
        self.model.L.weight.requires_grad = True
        #self.set_optimizer = torch.optim.Adam([{'params': [self.model.backbone.pruning_mask]},  {'params': self.model.L.parameters()}], lr=self.lr)
        #self.set_optimizer = torch.optim.Adam(self.model.backbone.pruning_mask, lr=self.lr)
        self.set_optimizer = torch.optim.Adam([
        {'params': [self.model.backbone.pruning_mask], 'lr': self.lr},  # Learning rate for pruning_mask
        {'params': self.model.L.parameters(), 'lr': 0.1}    ])



    def loop(self, support_size, support_images, support_labels, model, set_optimizer, backbone_grad=False):
        # Randomly shuffle support set
        rand_id = torch.randperm(support_size)
        selected_id = rand_id[:min(self.batch_size, support_size)]
        
        # Prepare batches
        train_batch = support_images[selected_id].to('cuda').squeeze()
        label_batch = support_labels[selected_id].to('cuda').squeeze()

        # Initialize model and optimizers only once
        way = torch.max(support_labels).item() + 1


        # Initialize the loss variable
        loss = None
        

        set_optimizer.zero_grad()
        logits = self.model(train_batch, backbone_grad=backbone_grad)
        #print('model grad before step mask',model.backbone.pruning_mask)

        loss = F.cross_entropy(logits, label_batch) #+ 0.005 * torch.norm(model.backbone.pruning_mask, 1)

        loss.backward()
        #print('model grad after backward mask',model.backbone.pruning_mask.grad)
        set_optimizer.step()
        # Clamp pruning_mask between 0 and 1
        with torch.no_grad():
            model.backbone.pruning_mask.clamp_(0, 1)

            #temp = 0.5
            #model.backbone.pruning_mask.data = torch.sigmoid(model.backbone.pruning_mask.data / temp)
            # Forward pass: Soft mask with sigmoid
            #mask = torch.sigmoid(model.backbone.pruning_mask / temp)

            # Binarize the mask using a straight-through estimator: Values greater than 0.5 become 1, others 0
            #binary_mask = (model.backbone.pruning_mask > 0.5).float()

            # Update pruning_mask to be binary in the forward pass
            #model.backbone.pruning_mask.data = steep_sigmoid(model.backbone.pruning_mask.data)

        
        #print('after step mask',(model.backbone.pruning_mask>0.5)*1)
        torch.set_printoptions(precision=2, sci_mode=False)
        print('after step mask',model.backbone.pruning_mask)
        print('binary mask \n',(model.backbone.pruning_mask>0.3)*1)
        print('average pruning level', f'{1-torch.mean(model.backbone.pruning_mask).item():.3f}')
        #print('model grad after step mask',model.backbone.pruning_mask.grad)
        # Compute accuracy
        logits = model(train_batch, backbone_grad=backbone_grad)  # Re-evaluate to get the logits for accuracy
        acc = accuracy(logits, label_batch)[0].item()
        
        return loss.item(), acc

    def test_forward(self, images, labels, dataset_index) -> Tensor:
        # turn backbone_grad off if backbone is not to be finetuned

        way = torch.max(labels[0]['support']).item()+1
        self.initialize_model(way)
        self.logreg = Logistic_Regression()


        with torch.no_grad():
            support_feat = self.model.backbone(images[0]['support'].squeeze().cuda())
        classi_score = self.logreg.forward(query_images = support_feat, support_images = support_feat , support_labels = labels[0]['support'])
        self.model.L.weight = torch.nn.Parameter(torch.tensor(self.logreg.classifier.coef_, dtype=torch.float32).to(self.model.L.weight.device), requires_grad=True)
        self.model.L.bias = torch.nn.Parameter(torch.tensor(self.logreg.classifier.intercept_, dtype=torch.float32).to(self.model.L.bias.device), requires_grad=True)

        total_loss = 0.0
        total_acc = 0.0
        step = 0
        epoch=0
        with torch.enable_grad():
            while epoch < self.ft_epoch2 and torch.sum(self.model.backbone.pruning_mask.data<0.3)<5:
                print('### \n',torch.sum(self.model.backbone.pruning_mask.data<0.3) )
                loss, acc = self.loop(support_size = labels[0]['support'].shape[0],support_images = images[0]['support'] ,support_labels = labels[0]['support'],model=self.model,set_optimizer= self.set_optimizer, backbone_grad=True)
                total_loss += loss
                total_acc += acc

                step += 1
                print(f'{step}, {total_loss/step:.2f}, {total_acc/step:.2f}')
                epoch+=1

        print('training over')
        self.model.backbone.pruning_mask.data = (self.model.backbone.pruning_mask.data>0.3)*1.0
        with torch.no_grad():
            support_feat = self.model.backbone(images[0]['support'].squeeze().cuda())
        classi_score = self.logreg.forward(query_images = support_feat, support_images = support_feat , support_labels = labels[0]['support'])
        self.model.L.weight = torch.nn.Parameter(torch.tensor(self.logreg.classifier.coef_, dtype=torch.float32).to(self.model.L.weight.device), requires_grad=True)
        self.model.L.bias = torch.nn.Parameter(torch.tensor(self.logreg.classifier.intercept_, dtype=torch.float32).to(self.model.L.bias.device), requires_grad=True)

        #self.model.backbone.pruning_mask.data = torch.ones((12,12)).cuda()*1.0
        print('####################')
        query_images = images[0]['query'].squeeze().cuda()
        batch_size = 250
        query_classi_list = []

        with torch.no_grad():
            for i in range(0, query_images.size(0), batch_size):
                # Select the batch of images
                batch = query_images[i:i+batch_size].squeeze().cuda()
                batch_classi_score = self.model(batch,backbone_grad = False)
                query_classi_list.append(batch_classi_score)
                

        query_classi_score = torch.cat(query_classi_list, dim=0)
        loss = F.cross_entropy(query_classi_score, labels[0]['query'].cuda().squeeze())
        acc = accuracy(query_classi_score, labels[0]['query'].cuda().squeeze())
        return loss , acc 


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




def get_model(config):
    return Mask(config)