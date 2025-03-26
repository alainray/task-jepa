import lightning as L
import torch
import torch.nn as nn
from models import VisionTransformer,LatentVisionTransformer, MultiHeadClassifier, PairVisionTransformer, MultiHeadClassifier
from lightning.pytorch.callbacks import ModelCheckpoint
from rich.table import Table
from rich.console import Console
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
import torch.nn.functional as F
from datasets import get_dataloaders
import wandb
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import torchmetrics
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from models import create_model
from utils import get_exp_name


class LightningRepClassification(L.LightningModule):
    def __init__(self, args, encoder, modulator):
        super().__init__()
        self.encoder = encoder
        self.use_reps = encoder is None
        self.modulator = modulator
        self.criterion = F.cosine_similarity
        self.args = args
    
    # We try to replicate the learned representation as a start.
    # We learn to replicate the representation when latents are zero
    # and also learn to replicate it when latents are different to zero.
    def training_step(self, batch, batch_idx):
        
        split = "train"
        metrics = self.split_step(batch)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict({k: v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['train_loss'] # so lightning can train
    

    def get_metrics(self, data):
        
        metrics = dict()
        loss = 0
        if "same" ==  self.args.losses or "all" == self.args.losses:
            same_loss = 1 - self.criterion(data['mid_reps'], data['rep_tgt']).mean()
            loss += same_loss
            metrics['same_loss'] = same_loss

        if "class" == self.args.losses or "all" == self.args.losses:
            class_loss = F.cross_entropy(data['logits'], data['class_tgt'], reduction="mean")
            loss +=  class_loss
            metrics['class_loss'] = class_loss

            preds = data['logits'].argmax(dim=-1).view(-1)
            correct = (preds == data['class_tgt']).view(-1).float()

            accuracy = correct.sum()/correct.numel()
            metrics['class_acc'] = accuracy
            dtype=correct.dtype
            device=correct.device
            tasks = data['tasks'].view(-1)
            n_attrs = 5 if self.args.dataset == "idsprites" else 6
            #print(data['tasks'].shape,correct.shape)
            sum_per_group = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                                tasks,
                                                                                correct,
                                                                                reduce="sum")


            counts = torch.zeros(n_attrs, dtype=tasks.dtype, device=device).scatter_reduce(0, tasks, torch.ones_like(tasks).cuda(), reduce="sum")
            mean_per_group = sum_per_group/counts
            if self.args.dataset == "idsprites":
                task_names = ['shape','scale','orientation','x','y']
            else:
                task_names =['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
            for i, task in enumerate(task_names):
                metrics[f'class_{task}'] = mean_per_group[i]
            
        metrics['loss'] = loss

        #for k, v in metrics.items():
        #    metrics[k] = v.item()
        return metrics 

    def split_step(self, batch):    

        src_img, src_rep, imgs, gt_reps, latents = batch
        zero_latents = torch.zeros_like(latents)
        deltas = latents.sum(dim=-1)
        bs, n_classes, c, h, w = imgs.shape
        if self.args.encoder.arch == "cnn":
            imgs = imgs.view(bs*n_classes, c, h, w)
        src_rep = src_rep if self.use_reps else self.encoder(src_img.float())     # Image encoding
        mid_reps = gt_reps if self.use_reps else self.encoder(imgs.float())     # Image encoding
        if self.args.encoder.arch == "cnn":
            mid_reps = mid_reps.view(bs, n_classes, -1)

        src_rep = src_rep.unsqueeze(1).repeat((1,n_classes,1))
        reps = self.modulator(src_rep, latents)                        # predicted reps given latents
        reps = torch.nn.functional.normalize(reps, p=2.0, dim=1, eps=1e-12)
        tgt_reps = self.modulator(mid_reps, zero_latents)               # reps we are trying to achieve
        tgt_reps = torch.nn.functional.normalize(tgt_reps, p=2.0, dim=1, eps=1e-12)
        logits = torch.matmul(reps, tgt_reps.transpose(1,2)).view(-1, n_classes) # bs x 10 x 10 --> 10bs x 10
        data = dict()
        data['mid_reps'] = mid_reps
        data['rep_tgt'] = gt_reps
        data['logits'] = logits
        targets = torch.tensor(bs*list(range(n_classes))).view(-1, n_classes).to(logits.device)
        tasks = latents.abs().argmax(dim=-1)
        data['class_tgt'] = targets.view(-1)
        data['tasks'] = tasks
        metrics = self.get_metrics(data)

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = "val"
        metrics = self.split_step(batch)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['val_loss']
        

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        split = "test"
        metrics = self.split_step(batch)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['test_loss']

    def configure_optimizers(self):
        if self.encoder is None:
            param_groups = [
            {
                'params': self.modulator.parameters()}
            ]
        else:
            param_groups = [
                {'params': self.encoder.parameters(),
                 'params': self.modulator.parameters()}
            ]
        return torch.optim.AdamW(param_groups, lr=self.args.lr)

def create_loggers(args):
    loggers = []
    try:
        wandb_logger = WandbLogger(
            project='task_jepa',
            name="new_experiment",
            version=args.experiment_id,
        )
        wandb_logger.log_hyperparams(args)
        args.experiment_id = wandb_logger.version
        loggers.append(wandb_logger)
    except wandb.errors.CommError as e:
        print(e)
        pass
    
    '''output_path = f"results/{args.dataset}"
    csv_logger = CSVLogger(
        output_path,
        version = "v1",
        name=args.experiment_id
    )
    csv_logger.log_hyperparams(args)
    loggers.append(csv_logger)'''
    
    return loggers
    
def create_callbacks(args):
    callbacks = []
    save_top_k=-1
    dir_path = f"results/{args.dataset}/{args.experiment_id}"
    callbacks.append(ModelCheckpoint(dirpath=dir_path,
                                 filename="{step}",
                                 save_top_k=save_top_k,
                                 monitor="train_loss",
                                 every_n_train_steps=args.num_steps//5,
                                 every_n_epochs=None,
                                 save_last=True))
    return callbacks


def get_lightning_model(args):
    encoder, modulator = create_model(args)
    
    if args.train_method == "encoder_erm":
        model = LightningEncoderERM(args, encoder,modulator)
    
    elif args.train_method == "task_jepa":
        model = LightningTaskJEPA(args, encoder, modulator)

    elif args.train_method == "rep_train":
        model = LightningRepClassification(args, encoder, modulator)
    else:
        pass
        
    return model

def add_EMA_args(args):
    
    if args.train_method in ['task_jepa', 'ijepa']:
        
        args.ema = [args.ema_start, 1.0]      # exponential moving average
        args.ipe = len(dls['train']) # iterations per epoch
        args.momentum_scheduler = [args.ema[0] + i*(args.ema[1]-args.ema[0])/(args.ipe*args.num_epochs*args.ipe_scale/args.iters_per_ema)
                          for i in range(int(args.ipe*args.num_epochs*args.ipe_scale)+1)]
        args.momentum_scheduler = iter(args.momentum_scheduler)
    return args
    
def train(args):
    print("Loading dataloaders...", flush=True)
    dls = get_dataloaders(args)
    loggers = create_loggers(args)
    args.experiment_name = get_exp_name(args)
    callbacks = create_callbacks(args)
    wandb.run.name = args.experiment_name
    print("Creating model!", flush=True)
    model = get_lightning_model(args)
    print(model.modulator)
    ckpt_path = None
    if args.resume_id:
        ckpt_path = f"results/{args.dataset}/{args.experiment_id}/last.ckpt" if args.resume_id else None
        print(f"Resuming from checkpoint at: {ckpt_path}", flush=True)
    args = add_EMA_args(args)
    
    trainer = L.Trainer(accelerator="gpu",
                        devices=1,
                        #max_epochs=args.num_epochs,
                        check_val_every_n_epoch=None,
                        max_steps=args.num_steps,
                        val_check_interval=args.num_steps//5,
                        callbacks=callbacks,
                        logger=loggers)
    trainer.fit(model=model,
                train_dataloaders=dls['train'],
                val_dataloaders=[dls['val']],
                ckpt_path = ckpt_path)
    
    
    wandb.finish()