"""
Module for episodic (meta) training/testing
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class EpisodicTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)
        self.classifier = get_classifier(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
    
    def forward(self, img_tasks, label_tasks, dataset_index , micro_batch_size=500, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        acc = []
        # Since batch_size = 1, we iterate over img_tasks directly
        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            total_query_loss = 0.
            total_query_acc = []
            num_queries = len(img_task["query"])
            for j in range(0, num_queries, micro_batch_size):
                # Micro-batch the query set
                query_batch = img_task["query"][j:j + micro_batch_size].squeeze_().cuda()
                query_labels_batch = label_tasks[i]["query"][j:j + micro_batch_size].squeeze_().cuda()

                query_features = self.backbone(query_batch)
                score = self.classifier(query_features, support_features,
                                        label_tasks[i]["support"].squeeze_().cuda(), **kwargs)

                # Compute loss and accuracy for the micro-batch
                total_query_loss += F.cross_entropy(score, query_labels_batch)
                total_query_acc.append(accuracy(score, query_labels_batch)[0])

            # Average the loss and accuracy over the micro-batches
            avg_query_loss = total_query_loss / micro_batch_size
            avg_query_acc = sum(total_query_acc) / len(total_query_acc)
            
            loss += avg_query_loss
            acc.append(avg_query_acc)

        loss /= batch_size  # Since batch_size is 1, this is effectively the final loss.
        return loss, acc
    
    def train_forward(self, img_tasks,label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks, *args, **kwargs)
    
    def val_forward(self, img_tasks,label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks, *args, **kwargs)
    
    def test_forward(self, img_tasks,label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks, *args, **kwargs)

def get_model(config):
    return EpisodicTraining(config)