import lightning as L
import torch
import torch.nn as nn
from models import get_lightning_model
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
    print(model.regressor)
    

    loss_dict = {'rep_train': ['class'], 
                 'rep_train_plus': ['class', 'non_mod_reg','mod_reg'],
                 'rep_train_same': ['same', 'non_mod_reg','mod_reg'],
                 'rep_train_plus_res': ['class', 'non_mod_reg','mod_reg'],
                 'rep_train_plus_film': ['class', 'non_mod_reg','mod_reg'],
                 'rep_train_same_res': ['same', 'non_mod_reg','mod_reg'],
                 'rep_train_same_film': ['same', 'non_mod_reg','mod_reg'],
                 'rep_train_plus_trans': ['class', 'non_mod_reg','mod_reg'],
                 'rep_train_same_trans': ['same', 'non_mod_reg','mod_reg'],
                 'rep_train_same_linop': ['same', 'non_mod_reg','mod_reg'],
                 'rep_train_same_latdir': ['same', 'non_mod_reg','mod_reg', "orth"],
                 'non_mod_regression': ['non_mod_reg'],
                 'mod_regression': ['non_mod_reg', 'mod_reg'],
                 'mod_regression_trans': ['non_mod_reg', 'mod_reg'],
                 'mod_regression_film': ['non_mod_reg', 'mod_reg'],
                 'mod_regression_linop': ['non_mod_reg', 'mod_reg'],
                 'mod_regression_latdir': ['non_mod_reg', 'mod_reg'],
                 'regression': ['non_mod_reg']
                 }
    args.losses = loss_dict[args.train_method]
    ckpt_path = None
    if args.resume_id:
        ckpt_path = f"results/{args.dataset}/{args.experiment_id}/last.ckpt" if args.resume_id else None
        print(f"Resuming from checkpoint at: {ckpt_path}", flush=True)
    args = add_EMA_args(args)
    
    trainer = L.Trainer(accelerator="gpu",
                        devices=1,
                        enable_progress_bar=False,
                        check_val_every_n_epoch=None,
                        max_steps=args.num_steps,
                        val_check_interval=50000,
                        callbacks=callbacks,
                        logger=loggers)
    trainer.fit(model=model,
                train_dataloaders=dls['train'],
                val_dataloaders=[dls['val']],
                ckpt_path = ckpt_path)
    
    
    wandb.finish()