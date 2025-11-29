import os
import pprint
from dataloader import build_HDF5_feat_dataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from utils import Struct, MetricLogger, accuracy
from model import ABMIL
from torch import nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, criterion, data_loader, optimizer,
                    device):
    model.train()
    #Added for wandb     
    running_loss = 0.0
    total = 0
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32)
        batch_size = bag.shape[0]
        label = data['label'].to(device)
        train_logits = model(bag)
        train_loss = criterion(train_logits.view(batch_size, -1), label)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # added for wandb
        running_loss += train_loss.item() * batch_size
        total += batch_size
    # return so wandb can read the running loss
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf):
    model.eval()
    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter="  ")
    for data in data_loader:
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        slide_preds = model(image_patches)
        loss = criterion(slide_preds, labels)
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
        #c.update(vars(args))
        #weifeng added to allow command line override of config file
        wandb_keys = {'wandb', 'wandb_project', 'wandb_entity', 'wandb_run_name'}
        override_args = {
            k: v for k, v in vars(args).items()
            if v is not None and k not in wandb_keys
        }
        c.update(override_args)
        print("Used config:", c, flush=True)
        conf = Struct(**c)
    
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf=conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    
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
    criterion = nn.CrossEntropyLoss()
    #weifeng add for wandb
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=conf.__dict__
        )
        wandb.watch(model, log_freq=100)

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):
        #train_one_epoch(model, criterion, train_loader, optimizer,
        #                device)
        # weifeng modified to get train_loss for wandb
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device
        )
        val_auc, val_acc, val_f1, val_loss = evaluate(model, criterion, val_loader, device, conf)
        test_auc, test_acc, test_f1, test_loss = evaluate(model, criterion, test_loader, device, conf)
        scheduler.step()
        if wandb_run:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/auc': val_auc,
                'val/acc': val_acc,
                'val/f1': val_f1,
                'val/loss': val_loss,
                'test/auc': test_auc,
                'test/acc': test_acc,
                'test/f1': test_f1,
                'test/loss': test_loss,
                'lr': optimizer.param_groups[0]['lr'],
            })
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
    # weifeng added to log best results to wandb
    if wandb_run:
        wandb_run.summary['best/epoch'] = best_state['epoch']
        wandb_run.summary['best/val_auc'] = best_state['val_auc']
        wandb_run.summary['best/val_acc'] = best_state['val_acc']
        wandb_run.summary['best/val_f1'] = best_state['val_f1']
        wandb_run.summary['best/test_auc'] = best_state['test_auc']
        wandb_run.summary['best/test_acc'] = best_state['test_acc']
        wandb_run.summary['best/test_f1'] = best_state['test_f1']
        wandb.finish()

def get_arguments():
    #parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser = argparse.ArgumentParser(
        'Patch classification training',
        add_help=True,
        allow_abbrev=False
    )
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml',
                        help='settings of Tip-Adapter in yaml format')
    #  weifeng added for wandb and attn_heads                    
    parser.add_argument('--attn_heads', '--attn_head', dest='attn_heads', type=int, default=None,
                        help='number of attention heads to override the config (default: use config value)')
    parser.add_argument('--wandb', action='store_true',
                        help='enable Weights & Biases logging')
    parser.add_argument('--wandb_project', '--wandb-project', default='ece1512-mil',
                        help='Weights & Biases project name (default: ece1512-mil)')
    parser.add_argument('--wandb_entity', '--wandb-entity', default=None,
                        help='Weights & Biases entity/team (optional)')
    parser.add_argument('--wandb_run_name', '--wandb-run-name', default=None,
                        help='Optional custom run name for Weights & Biases')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    main(args)