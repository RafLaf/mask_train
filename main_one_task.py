import sys

sys.path.append('/gpfsdswork/projects/rech/csb/ude77uf/lab/lib/python3.10/site-packages')

print(sys.path)
import argparse
from config import get_config
import os
from logger import create_logger
from data import create_torch_dataloader
from data.dataset_spec import Split
import torch
import numpy as np
import random
import json
from utils import accuracy, AverageMeter, delete_checkpoint, save_checkpoint, load_pretrained, auto_resume_helper, load_checkpoint
import torch
import datetime
from models import get_model
from optimizer import build_optimizer, build_scheduler
import time
import math
import sys
from time import sleep
import copy
import time


from torch.utils.tensorboard import SummaryWriter
import gc

def setup_seed(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_option():
    parser = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--train_batch_size', type=int, help="training batch size for single GPU")
    parser.add_argument('--valid_batch_size', type=int, help="validation batch size for single GPU")
    parser.add_argument('--test_batch_size', type=int, help="test batch size for single GPU")
    parser.add_argument('--output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--is_train', type=int, choices=[0, 1], help="training or testing")
    parser.add_argument('--pretrained', type=str, help="pretrained path") 
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--resume', help='resume path')
    parser.add_argument('--save-stats', help='path to save test stats')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def train(config):
    train_dataloader, train_dataset  = create_torch_dataloader(Split.TRAIN, config)
    valid_dataloader, valid_dataset = create_torch_dataloader(Split.VALID, config)
    writer = SummaryWriter(log_dir=config.OUTPUT)

    num_classes = train_dataset.num_classes if hasattr(train_dataset, "num_classes") else None

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    
    model = get_model(config, num_classes).cuda() if num_classes is not None else get_model(config).cuda()

    max_accuracy = [0.0]*config.SAVE_TOP_K_MODEL

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_dataloader))

    step = 0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy, step = load_checkpoint(config, model, optimizer, lr_scheduler, logger)

    if config.MODEL.PRETRAINED:
        load_pretrained(config, model, logger)
        acc1, loss = validate(config, valid_dataset, valid_dataloader, model)
        logger.info(f"Accuracy of the network on the {len(valid_dataloader)} test images: {acc1:.1f}%")

    


    

    logger.info("Start training")
    start_time = time.time()

    

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        step = train_one_epoch(config, model, train_dataset, train_dataloader, optimizer, epoch, lr_scheduler, step, 
                             writer)
        acc_current, loss = validate(config, valid_dataset, valid_dataloader, model, epoch, writer)
        logger.info(f"Accuracy of the network on the validated images: {acc_current:.1f}%")

        # is current accuracy in topK?
        topK = None
        for i, acc in enumerate(max_accuracy):
            if acc_current > acc:
                max_accuracy.insert(i, acc_current)
                max_accuracy.pop()
                topK = i+1
                break

        # if current accuracy is in topK, delete the worst checkpoint
        if topK is not None:
            delete_checkpoint(config, topK)

        # delete previous checkpoint
        if epoch-1 not in config.SAVE_EPOCHS:
            delete_checkpoint(config, epoch=epoch-1)
            
        save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler,
                    logger, topK, step)
            
        logger.info(f'Max Top {config.SAVE_TOP_K_MODEL} accuracy: {max_accuracy}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def save_results(config, acc1, ci, prunning_matrix,heuristic=False, task_number=None):
    #path = os.path.join(config.OUTPUT, f"test/results_lora_{config.MODEL.OPTIMIZER_HEAD}_{config.DATA.TEST.DATASET_NAMES[0]}_{config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT}_shots_task{task_number}.pt")
    path = os.path.join(config.OUTPUT, f"results_{config.MODEL.TYPE}_{config.MODEL.OPTIMIZER_HEAD}_{config.DATA.TEST.DATASET_NAMES[0]}_{config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT}_shots_task{task_number}.pt")
    if not os.path.exists(path):
        result_dic = {}
        result_dic[f"{config.DATA.TEST.DATASET_NAMES[0]}"] = []
        result_dic[f"heuristic_{config.DATA.TEST.DATASET_NAMES[0]}"] = []
    else:
        result_dic = torch.load(path)
    if f"{config.DATA.TEST.DATASET_NAMES[0]}" not in result_dic.keys():
        result_dic[f"{config.DATA.TEST.DATASET_NAMES[0]}"] = []
    if f"heuristic_{config.DATA.TEST.DATASET_NAMES[0]}" not in result_dic.keys():
        result_dic[f"heuristic_{config.DATA.TEST.DATASET_NAMES[0]}"] = []
    if heuristic:
        result_dic[f"heuristic_{config.DATA.TEST.DATASET_NAMES[0]}"].append([acc1, dict(config.MODEL) ,ci, prunning_matrix])
    else:
        result_dic[f"{config.DATA.TEST.DATASET_NAMES[0]}"].append([acc1,dict(config.MODEL) ,ci, prunning_matrix]) #here acc1 is -loss
    torch.save(result_dic, path)
    logger.info(f"Results saved to {path}")


def test(config, prunning_matrix=None, task_number=None , heuristic=False, pre_loaded_data=None):
    if pre_loaded_data is None:
        test_dataloader, test_dataset = create_torch_dataloader(Split.TEST, config)
    else:
        test_dataloader = '_'
        test_dataset = '_'
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = get_model(config).cuda()
    
    if config.MODEL.PRETRAINED:
        load_pretrained(config, model, logger)
    if prunning_matrix is not None:
        if config.MODEL.BACKBONE == 'clip':
            model.backbone.model.visual.apply_pruning_to_vit(prunning_matrix)
        if config.MODEL.BACKBONE == 'DINO_v2':
            model.backbone.apply_pruning_to_vit(prunning_matrix)
    # if model has adapters like in TSA
    #print(config.MODEL.CLASSIFIER_PARAMETERS)
    if config.MODEL.TYPE=='fewshot_finetune':
        condition  = config.MODEL.CLASSIFIER_PARAMETERS[-2] or config.MODEL.CLASSIFIER_PARAMETERS[-3]
    else:
        condition = False

    if hasattr(model, 'mode') and model.mode == "NCC" and condition:
        model.append_adapter()


    logger.info("Start testing")
    
    with torch.no_grad():
        acc1, loss, ci = testing(config, test_dataset, test_dataloader, model, task_number, heuristic=heuristic , pre_loaded_data=pre_loaded_data)
        if heuristic:
            logger.info(f"Test HEURISTIC Accuracy of {config.DATA.TEST.DATASET_NAMES[0]}: {acc1:.2f}%+-{ci:.2f}")
        else:
            logger.info(f"Test Accuracy of {config.DATA.TEST.DATASET_NAMES[0]}: {acc1:.2f}%+-{ci:.2f}")
    
    # logging testing results in config.OUTPUT/results.json
    if not heuristic:
        try:
            prunning_matrix = model.model.backbone.pruning_mask
        except:
            prunning_matrix = None
        save_results(config, acc1, ci, prunning_matrix=prunning_matrix, task_number=task_number)
    if heuristic:
        return -loss
    else:
        return acc1





def train_one_epoch(config, model, dataset, data_loader, optimizer, epoch, lr_scheduler, step, writer):
    model.train()
    optimizer.zero_grad()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    start = time.time()
    end = time.time()

    dataset.set_epoch()
    pathss = []
    all_label = []
    for idx, batches in enumerate(data_loader):
        dataset_index, imgs, labels = batches
        loss, acc = model.train_forward(imgs, labels, dataset_index)
        acc = torch.mean(torch.stack(acc))
        loss.backward()
        optimizer.step()
        if config.TRAIN.SCHEDULE_PER_STEP:
            lr_scheduler.step_update(step)
            step += 1
        optimizer.zero_grad()
        loss_meter.update(loss.item())
        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("Acc/train", acc.item(), step)
        acc_meter.update(acc.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']      
            writer.add_scalar("lr", lr, step)    
            if not ((not config.DATA.TRAIN.IS_EPISODIC) and config.DATA.TRAIN.ITERATION_PER_EPOCH is None and len(config.DATA.TRAIN.DATASET_NAMES) > 1):
                etas = batch_time.avg * (len(data_loader) - idx-1)
                logger.info(
                    f'time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    f'loss {loss_meter.val:.2f} ({loss_meter.avg:.2f})\t'
                    f'Train: [{epoch+1}/{config.TRAIN.EPOCHS}][{idx+1}/{len(data_loader)}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t')
            else:
                logger.info(
                    f'time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    f'loss {loss_meter.val:.2f} ({loss_meter.avg:.2f})\t'
                    f'Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t')
    if not config.TRAIN.SCHEDULE_PER_STEP:
        lr_scheduler.step_update(step)
        step += 1
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    writer.add_scalar("Loss/train_epoch", loss_meter.avg, epoch)
    writer.add_scalar("Acc/train_epoch", acc_meter.avg, epoch)


    return step

@torch.no_grad()
def validate(config, dataset, data_loader, model, epoch=None, writer=None):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    end = time.time()

    dataset.set_epoch()

    for idx, batches in enumerate(data_loader):
        dataset_index, imgs, labels = batches
        
        
        loss, acc = model.val_forward(imgs, labels, dataset_index)
        
        acc = torch.mean(torch.stack(acc))

        loss_meter.update(loss.item())
        acc_meter.update(acc.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Val: [{idx+1}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                f'Loss {loss_meter.val:.2f} ({loss_meter.avg:.2f})\t'
                f'Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t')
    logger.info(f' * Acc@1 {acc_meter.avg:.2f}')
    if epoch is not None and writer is not None:
        writer.add_scalar("Loss/val_epoch", loss_meter.avg, epoch)
        writer.add_scalar("Acc/val_epoch", acc_meter.avg, epoch)
    return acc_meter.avg, loss_meter.avg   

@torch.no_grad()
def loo_classes(imgs, labels):
    '''leave on out in the support set, one sample of the support set is used as query set'''
    imgs = imgs[0]
    labels = labels[0]
    classes = torch.unique(labels['support'])
    imgs['query'] = torch.tensor([]).reshape(imgs['query'].shape[0],0,imgs['query'].shape[2], imgs['query'].shape[3], imgs['query'].shape[4])
    labels['query'] = torch.tensor([]).reshape(labels['support'].shape[0],0,labels['support'].shape[2])
    for i in classes:
        idx = torch.where(labels['support']==i)[1]
        removed =  idx[torch.randperm(len(idx))[0]]
        imgs['query'] = torch.cat((imgs['query'][0], imgs['support'][0,removed].unsqueeze(0)), dim=0).unsqueeze(0)
        imgs['support'] = torch.cat((imgs['support'][0,:removed], imgs['support'][0,removed+1:]), dim=0).unsqueeze(0)
        labels['query'] = torch.cat((labels['query'][0], labels['support'][0,removed].unsqueeze(0)), dim=0).unsqueeze(0).type(torch.LongTensor)
        labels['support'] = torch.cat((labels['support'][0,:removed], labels['support'][0,removed+1:]), dim=0).unsqueeze(0).type(torch.LongTensor)
    return [imgs], [labels]

@torch.no_grad()
def loo(imgs, labels,removed):
    '''leave one out in the support set, one sample of the support set is used as query set'''
    imgs_0 = imgs[0]
    labels_0 = labels[0]
    imgs_out = {}
    labels_out = {}
    imgs_out['query'] = torch.tensor([]).reshape(imgs_0['query'].shape[0],0,imgs_0['query'].shape[2], imgs_0['query'].shape[3], imgs_0['query'].shape[4])
    labels_out['query'] = torch.tensor([]).reshape(labels_0['support'].shape[0],0,labels_0['support'].shape[2])
    
    imgs_out['support'] = torch.cat((imgs_0['support'][0,:removed], imgs_0['support'][0,removed+1:]), dim=0).unsqueeze(0)
    labels_out['support'] = torch.cat((labels_0['support'][0,:removed], labels_0['support'][0,removed+1:]), dim=0).unsqueeze(0).type(torch.LongTensor)
    for _ in range(2): #doing it twice for a batch of size 2 so that forard does not squeeze btach dim and fail!
        imgs_out['query'] = torch.cat((imgs_out['query'][0], imgs_0['support'][0,removed].unsqueeze(0)), dim=0).unsqueeze(0)
        labels_out['query'] = torch.cat((labels_out['query'][0], labels_0['support'][0,removed].unsqueeze(0)), dim=0).unsqueeze(0).type(torch.LongTensor)

    return [imgs_out], [labels_out]




def soft_ncm(imgs,labels,model):
    # find the centroid for neasrest class mean torch.Size([1, 24, 3, 224, 224]) torch.Size([1, 24, 1] #torch.Size([1, 2, 3, 224, 224]) torch.Size([1, 2, 1])

    # find the centroid for neasrest class mean torch.Size([1, 24, 3, 224, 224]) torch.Size([1, 24, 1] #torch.Size([1, 2, 3, 224, 224]) torch.Size([1, 2, 1])
    features_support = model.backbone.model.visual(imgs[0]['support'].squeeze_().cuda())
    features_query = model.backbone.model.visual(imgs[0]['query'].squeeze_().cuda())
    centroids = []
    for i in torch.unique(labels[0]['support']):
        idx = torch.where(labels[0]['support']==i)[1]
        centroids.append(torch.mean(features_support[idx],dim=0))
    centroids = torch.stack(centroids)
    # find the distance between the query and the centroids
    dist = torch.cdist(features_query,centroids)
    # find the softmax of the distance
    dist= torch.nn.functional.softmax(-dist, dim=1)[0]
    # find the proototypical loss
    labels_query = int(labels[0]['query'][0,0,0].detach())

    return dist[labels_query] , [torch.tensor([float(torch.argmax(dist)==labels_query)*100])]

    

    



@torch.no_grad()
def testing(config, dataset,data_loader, model, task_number=None, heuristic=False, pre_loaded_data=None):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    end = time.time()
    accs = []

    
    
    if pre_loaded_data is not None:
        dataset_index, imgs, labels = pre_loaded_data
    else:
        dataset.set_epoch()
        for idx, batches in enumerate(data_loader):
            if idx == task_number:
                dataset_index, imgs, labels = batches
                break
    
    loss, acc = model.test_forward(imgs, labels, dataset_index)
    print(loss, acc)
    accs.extend(acc)
    acc = torch.mean(torch.stack(acc))

    loss_meter.update(loss.item())
    acc_meter.update(acc.item())

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    #if idx % config.PRINT_FREQ == 0:
    #    logger.info(
    #        f'Test: [{idx+1}/{len(data_loader)}]\t'
    #        f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
    #        f'Loss {loss_meter.val:.2f} ({loss_meter.avg:.2f})\t'
    #        f'Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t')

    accs = torch.stack(accs)
    print(accs)
    if args.save_stats and not heuristic:        
        torch.save(accs, args.save_stats)
    
    ci = (1.96*torch.std(accs)/math.sqrt(accs.shape[0])).item()  #should be student's t distribution
    return acc_meter.avg, loss_meter.avg, ci

if __name__ == '__main__':
    args, config = parse_option()
    torch.cuda.set_device(config.GPU_ID)
    

    config.defrost()

    config.freeze()

    setup_seed(config.SEED)
    
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")


    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    
    if config.IS_TRAIN:
        train(config)
    else:
        start = time.time()
        topK = 1
        TN=int(sys.argv[3])
        print('task_number chosen:', TN)
        sleep(1)
        dataset_loader = create_torch_dataloader(Split.TEST, config)
        #for idx, batches in enumerate(dataset_loader[1]):
        #    print('hello')
        #    if idx == TN:
        #        data = batches
        #        break
        data = dataset_loader[1][TN]
        print('data loaded')
        dataset_loader = 0
        # Run garbage collector
        gc.collect()
        data_copy = copy.deepcopy(data)
        start_acc = test(config, prunning_matrix=None,task_number=TN, pre_loaded_data=data_copy)
        logger.info(f"Best accuracy: {start_acc}")
        end = time.time()
        print('time taken:', end-start)


