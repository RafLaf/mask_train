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
        head: Use fc head or PN head to finetune.
        '''
        self.backbone = deepcopy(backbone)
        if self.backbone.name == 'clip':
            target_modules = ['in_proj_weight']
        if self.backbone.name == 'dinov2':
            target_modules = ['qkv']
        self.config_lora = LoraConfig(
                        r=16,
                        lora_alpha=16,
                        target_modules=target_modules,
                        lora_dropout=0.1,
                        bias="none"         )
        
        if self.backbone.name == 'dinov2':
            self.backbone.model = get_peft_model(self.backbone.model, self.config_lora)
        if self.backbone.name == 'clip':
            self.backbone.model.visual.transformer.resblocks = get_peft_model(self.backbone.model.visual.transformer.resblocks, self.config_lora)
        self.L = nn.Linear(backbone.outdim, way).to(device)


    def forward(self, x, backbone_grad = True):
        # turn backbone_grad off if backbone is not to be finetuned
        if backbone_grad:
            x = self.backbone(x)
        else:
            with torch.no_grad():
                x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.L(x)
        return x

class LORAtuner(nn.Module):
    def __init__(self, config):
        '''
        backbone: the pre-trained backbone
        ft_batchsize: batch size for finetune
        feed_query_batchsize: max number of query images processed once (avoid memory issues)
        ft_epoch: epoch of finetune
        '''
        super().__init__()
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)
        self.ft_batchsize = 128
        self.ft_epoch1 = config.MODEL.EPOCH_HEAD
        self.ft_epoch2 = config.MODEL.EPOCH_LORA
        self.lr1  = config.MODEL.LR_HEAD
        self.lr2 = config.MODEL.LR_LORA
        self.C_HEAD = config.MODEL.REGULARIZATION_HEAD
        self.C_LORA = config.MODEL.REGULARIZATION_LORA
        self.OPTIMIZER_HEAD = config.MODEL.OPTIMIZER_HEAD
        self.OPTIMIZER_LORA = config.MODEL.OPTIMIZER_LORA
        self.num_ways = config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_WAYS
        self.model = None  # Will hold the FinetuneModel instance



    
    def loop(self, support_size, support_images, support_labels, model, set_optimizer, backbone_grad=False):
        # Randomly shuffle support set
        rand_id = torch.randperm(support_size)
        selected_id = rand_id[:min(self.ft_batchsize, support_size)]
        
        # Prepare batches
        train_batch = support_images[selected_id].to('cuda').squeeze()
        label_batch = support_labels[selected_id].to('cuda').squeeze()


        # Initialize the loss variable
        loss = None
        
        # Define the closure for LBFGS
        def closure():
            nonlocal loss
            if torch.is_grad_enabled():
                set_optimizer.zero_grad()  # Clear previous gradients
            logits = model(train_batch, backbone_grad=backbone_grad)
            # Calculate the loss with L2 regularization
            l2_norm = torch.norm(model.L.weight, p=2)  # L2 norm of model's weights
            loss = F.cross_entropy(logits, label_batch) + self.C_HEAD * l2_norm
            loss.backward()  # Backpropagate the loss
            
            return loss

        # Forward pass and optimization
        if isinstance(set_optimizer, torch.optim.LBFGS):
            set_optimizer.step(closure)
            print(loss)
        else:
            set_optimizer.zero_grad()
            logits = model(train_batch, backbone_grad=backbone_grad)
            loss = F.cross_entropy(logits, label_batch)
            loss.backward()
            set_optimizer.step()
        
        # Compute accuracy
        logits = model(train_batch, backbone_grad=backbone_grad)  # Re-evaluate to get the logits for accuracy
        acc = accuracy(logits, label_batch)[0].item()
        
        return loss.item(), acc

    def test_forward(self, images, labels, dataset_index) -> Tensor:
        """Take one task of few-shot support examples and query examples as input,
            output the logits of each query examples.

        Args:
            query_images: query examples. size: [num_query, c, h, w]
            support_images: support examples. size: [num_support, c, h, w]
            support_labels: labels of support examples. size: [num_support, way]
        Output:
            classification_scores: The calculated logits of query examples.
                                   size: [num_query, way]
        """
        way = torch.max(labels[0]['support']).item()+1
        self.model = FinetuneModel(self.backbone, way, device = 'cuda')
        if self.OPTIMIZER_HEAD=='LBFGS':
            set_optimizer_1 = torch.optim.LBFGS(self.model.L.parameters(), max_iter = 1000)
        elif self.OPTIMIZER_HEAD == 'adam':
            set_optimizer_1 = torch.optim.Adam(self.model.L.parameters(), lr = self.lr1, weight_decay = self.C_HEAD)#, momentum=0.9)
        elif self.OPTIMIZER_HEAD == 'scikit':
            from architectures.classifier.LR import Logistic_Regression
            logreg = Logistic_Regression()
        else:
            raise NotImplementedError

        set_optimizer_2 = torch.optim.Adam(self.model.parameters(), lr = self.lr2,weight_decay = self.C_LORA)#, momentum=0.9)
        
        step = 0
        total_loss = 0.0
        total_acc = 0.0
        with torch.enable_grad():
            if self.OPTIMIZER_HEAD == 'scikit':
                    with torch.no_grad():
                        support_images = self.model.backbone(images[0]['support'].squeeze().cuda())
                    classi_score = logreg.forward(query_images = support_images, support_images = support_images , support_labels = labels[0]['support'])
                    self.model.L.weight = torch.nn.Parameter(torch.tensor(logreg.classifier.coef_, dtype=torch.float32).to(self.model.L.weight.device), requires_grad=True)
                    self.model.L.bias = torch.nn.Parameter(torch.tensor(logreg.classifier.intercept_, dtype=torch.float32).to(self.model.L.bias.device), requires_grad=True)
            else:
                for epoch in range(self.ft_epoch1):
                    loss ,acc = self.loop(support_size = labels[0]['support'].shape[0],support_images = images[0]['support'] ,support_labels = labels[0]['support'],model=self.model,set_optimizer= set_optimizer_1, backbone_grad=False)
                    total_loss += loss
                    total_acc += acc
                    step += 1
                    print(f'{step}, {total_loss/step:.2f}, {total_acc/step:.2f}')

        print('####################')
        with torch.enable_grad():
            for epoch in range(self.ft_epoch2):
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
                loss, acc = self.loop(support_size = labels[0]['support'].shape[0],support_images = images[0]['support'] ,support_labels = labels[0]['support'],model=self.model,set_optimizer= set_optimizer_2, backbone_grad=True)
                total_loss += loss
                total_acc += acc

                step += 1
                print(f'{step}, {total_loss/step:.2f}, {total_acc/step:.2f}')

        self.model.eval()    
        print('training over')
        print('####################')
        print('scikit_results') 
        query_images = images[0]['query'].squeeze().cuda()
        batch_size = 128
        query_classi_list = []

        with torch.no_grad():
            for i in range(0, query_images.size(0), batch_size):
                # Select the batch of images
                batch = query_images[i:i+batch_size].squeeze().cuda()
                batch_classi_score = self.model(batch,backbone_grad = False)
                query_classi_list.append(batch_classi_score)
                

        query_classi_score = torch.cat(query_classi_list, dim=0)

        
        #query_predictions = logreg.classifier.predict(query_features.detach().cpu().numpy())
        #accuracy_scikit = accuracy_score(labels[0]['query'], query_predictions)
        #print( accuracy_scikit)
        #print('####################')     
        #classification_scores = model(images[0]['query'].squeeze().cuda(), backbone_grad = False)
        loss = F.cross_entropy(query_classi_score, labels[0]['query'].cuda().squeeze())
        acc = accuracy(query_classi_score, labels[0]['query'].cuda().squeeze())
        return loss , acc 

def get_model(config):
    return LORAtuner(config)