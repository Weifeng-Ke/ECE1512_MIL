import os
import pprint
from dataloader import build_HDF5_feat_dataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from utils import Struct, MetricLogger, accuracy
from model import ABMIL, ContrastiveLossVectorized
from torch import nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np
from tqdm import tqdm
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' 
[Hsien, 30 Nov 25]
Add collate function to pad all bags to same length.
Add set-seed function for fair comparison.
'''
def mil_collate_fn(batch):
    # Batch: LIST of dataset samples {'input': ..., 'coords': ..., 'label': ...}

    # Determine max number of instances in this batch
    max_len = max(item['input'].shape[0] for item in batch)

    padded_feats = []
    padded_coords = []
    labels = []

    # Iterate through each bag in the batch and pad to same bag length (N)
    for item in batch:
        feat = item['input']
        coords = item['coords']
        label = item['label']

        N, D = feat.shape
        padded_feat = np.zeros((max_len, D), dtype=feat.dtype)
        padded_feat[:N] = feat
        
        padded_coord = np.zeros((max_len, coords.shape[1]), dtype=coords.dtype)
        padded_coord[:N] = coords

        padded_feats.append(padded_feat)
        padded_coords.append(padded_coord)
        labels.append(label)

    return {
        'input': torch.tensor(np.array(padded_feats)),                              # (B, max_len, D)
        'coords': torch.tensor(np.array(padded_coords)),                            # (B, max_len, 2)
        'mask': torch.tensor(
            [[1]*item['input'].shape[0] + [0]*(max_len - item['input'].shape[0])
            for item in batch]),                                                    # (B, max_len)
        'label': torch.tensor(labels)                                               # (B,)
    }

def set_seed(seed_value):
    random.seed(seed_value)         # Set random library seed
    np.random.seed(seed_value)      # Set numpy seed    
    torch.manual_seed(seed_value)   # Set torch seed
    if torch.cuda.is_available():                       # Set CUDA/GPU seed
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    # Set deterministic options for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash-based randomization (important for PyTorch DataLoader)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def train_one_epoch(model, 
                    main_criterion, 
                    contrastive_criterion,
                    data_loader, 
                    optimizer,
                    device,
                    alpha=1.0
                    ):
    model.train()
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32)
        batch_size = bag.shape[0]
        label = data['label'].to(device)
        mask = data['mask'].to(device)
        train_logits, instance_embeddings, bag_embedding = model(bag, mask=mask)

        # Instance Embeddings : (B, N, D_inner) whereby N is max across bag
        # Bag Embedding:      : (B, D_inner)

        # Cross Entropy loss
        ce_loss = main_criterion(train_logits.view(batch_size, -1), label)

        if (contrastive_criterion is not None):
            # Contrastive loss
            cont_loss = contrastive_criterion(instance_embeddings, bag_embedding, mask, device)

            # Combined Loss
            train_loss = (ce_loss * alpha) + (cont_loss * (1 - alpha))

        else:
            train_loss = ce_loss
        
        # print("Train Loss:", train_loss.item())
        # print(train_loss)
        if (torch.isnan(train_loss)):
            print("[!]NaN loss encountered, skipping update")
            continue
        # if (train_loss < 0):
        #     print("[!]Negative loss encountered")


        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, 
             main_criterion, 
             contrastive_criterion, 
             data_loader, 
             device, 
             conf, 
             alpha=1.0
             ):
    model.eval()
    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter="  ")
    for data in data_loader:
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        mask = data['mask'].to(device)
        slide_preds, instance_embeddings, bag_embedding = model(image_patches, mask=mask)
        
        # Compute main classification loss
        ce_loss = main_criterion(slide_preds, labels)

        # Compute contrastive loss
        if (contrastive_criterion is not None):
            cont_loss = contrastive_criterion(instance_embeddings, bag_embedding, mask, device)

            # Combine losses (weighted)
            loss = (ce_loss * alpha) + (cont_loss * (1 - alpha))
        else:
            loss = ce_loss
        
        pred = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(pred, labels, topk=(1,))[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])
        y_pred.append(pred)
        y_true.append(labels)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, task='multiclass').to(device)
    AUROC_metric(y_pred, y_true)
    auroc = AUROC_metric.compute().item()
    F1_metric = torchmetrics.F1Score(num_classes=conf.n_class, average='macro').to(device)
    F1_metric(y_pred, y_true)
    f1_score = F1_metric.compute().item()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg



def main(args):
    
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        print("Used config:", c, flush=True)
        conf = Struct(**c)
    

    set_seed(42)        # Set seed for reproducibility

    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf=conf)

    train_loader    = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                        num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True,
                        collate_fn=mil_collate_fn)
    val_loader      = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                        num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False,
                        collate_fn=mil_collate_fn)
    test_loader     = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                        num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False,
                        collate_fn=mil_collate_fn)
    
    # count = 0

    # for data in train_loader:
    #     bag = data['input']
    #     labels = data['label']
    #     count += 1
    #     print(f"bag.shape: {bag.shape}, labels: {labels}")
    #     if count > 3:
    #         break
    
    model = ABMIL(conf=conf, D=conf.D_feat)

    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=conf.lr, 
                                  weight_decay=conf.wd)
    
    sched_warmup = LinearLR(
        optimizer,
        start_factor=0.01,  # ~0.0 to avoid zero-LR edge cases
        end_factor=1.0,
        total_iters=conf.warmup_epoch
    )

    # Cosine from base_lr -> eta_min
    sched_cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, conf.train_epoch - conf.warmup_epoch),
        eta_min=1e-10
    )

    # Call warmup for first `warmup_epochs`, then cosine after
    scheduler = SequentialLR(
        optimizer,
        schedulers=[sched_warmup, sched_cosine],
        milestones=[conf.warmup_epoch]
    )

    # Display the configuration settings
    print('Configuration:')
    pprint.pprint(conf)
    ce_criterion            = nn.CrossEntropyLoss()
    if (conf.use_contrastive_loss):
        contrastive_criterion   = ContrastiveLossVectorized(temperature=conf.temperature)
    else:
        contrastive_criterion = None
    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):
        train_one_epoch(model, ce_criterion, contrastive_criterion, train_loader, optimizer, device, alpha=conf.alpha)
        
        val_auc, val_acc, val_f1, val_loss = evaluate(model, ce_criterion, contrastive_criterion, val_loader, device, conf, alpha=conf.alpha)
        test_auc, test_acc, test_f1, test_loss = evaluate(model, ce_criterion, contrastive_criterion, test_loader, device, conf, alpha=conf.alpha)
        print("------------------------------")
        scheduler.step()
        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1

    print("Results on best epoch:")
    print(best_state)

def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml',
                        help='settings of Tip-Adapter in yaml format')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    main(args)