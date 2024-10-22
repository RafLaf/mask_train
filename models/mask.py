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
        self.reg_mask = config.MODEL.REG_MASK
        self.batch_size = 128
        self.binary = config.MODEL.BINARY_MASK
        self.n_lowest = config.MODEL.N_LOWEST


    def initialize_model(self, way):
        '''
        Initializes the FinetuneModel if not already initialized or if the number of classes (way) changes.
        '''
        if self.model is None or self.model.L.out_features != way:
            self.model = FinetuneModel(self.backbone, way, device='cuda')

        self.L = nn.Linear(self.backbone.outdim, way)
        for param in self.model.parameters():
            param.requires_grad = False
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
            pass
            #model.backbone.pruning_mask.clamp_(0, 1)

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
            support_images = images[0]['support'].squeeze().cuda()  # Extract and move to GPU
            batches = torch.split(support_images, 2000)  # Split into batches of size 2000

            all_support_feat = []  # To collect all the features
            for batch in batches:
                support_feat = self.model.backbone(batch)  # Get features for each batch
                all_support_feat.append(support_feat)

            # Concatenate all the features back together
            support_feat = torch.cat(all_support_feat, dim=0)
        classi_score = self.logreg.forward(query_images = support_feat, support_images = support_feat , support_labels = labels[0]['support'])
        self.model.L.weight = torch.nn.Parameter(torch.tensor(self.logreg.classifier.coef_, dtype=torch.float32).to(self.model.L.weight.device), requires_grad=True)
        self.model.L.bias = torch.nn.Parameter(torch.tensor(self.logreg.classifier.intercept_, dtype=torch.float32).to(self.model.L.bias.device), requires_grad=True)

        total_loss = 0.0
        total_acc = 0.0
        step = 0
        epoch=0
        with torch.enable_grad():
            while epoch < self.ft_epoch2:# and torch.sum(self.model.backbone.pruning_mask.data<0.3)<2:
                print_params = True
                if print_params:
                    total_params = sum([p.numel() for p in self.model.parameters()])
                    trainable_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
                    print(
                    f"""
                    {total_params} total params,
                    {trainable_params}" trainable params,
                    {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
                    """
                    )
                loss, acc = self.loop(support_size = labels[0]['support'].shape[0],support_images = images[0]['support'] ,support_labels = labels[0]['support'],model=self.model,set_optimizer= self.set_optimizer, backbone_grad=True)
                total_loss += loss
                total_acc += acc

                step += 1
                print(f'{step}, {total_loss/step:.2f}, {total_acc/step:.2f}')
                epoch+=1

        print('training over')
        if self.binary:
            print('before', self.model.backbone.pruning_mask)
            def set_n_lowest_to_zero(tensor, n_lowest):
                num_rows, num_cols = tensor.shape

                # Flatten the tensor and get the indices of the N smallest values
                flattened_tensor = tensor.flatten()
                _, indices = torch.topk(flattened_tensor, k=n_lowest, largest=False)
                
                # Set the N lowest values to zero
                for idx in indices:
                    # Convert flattened index back to 2D index manually
                    row_idx = idx // num_cols
                    col_idx = idx % num_cols
                    print('row_idx',row_idx, 'col_idx',col_idx)
                    tensor[row_idx, col_idx] = 0

            pruning_mask = self.model.backbone.pruning_mask.data
            n_lowest = self.n_lowest

            # Set N lowest values in the tensor to 0
            set_n_lowest_to_zero(pruning_mask, n_lowest)
            print('after', self.model.backbone.pruning_mask)
        


        with torch.no_grad():
            support_images = images[0]['support'].squeeze().cuda()  # Extract and move to GPU
            batches = torch.split(support_images, 2000)  # Split into batches of size 2000

            all_support_feat = []  # To collect all the features
            for batch in batches:
                support_feat = self.model.backbone(batch)  # Get features for each batch
                all_support_feat.append(support_feat)

            # Concatenate all the features back together
            support_feat = torch.cat(all_support_feat, dim=0)
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