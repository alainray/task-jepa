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

def fig_to_img(fig):
    # Attach the canvas to the figure
    canvas = FigureCanvas(fig)

    # Render the figure into a buffer
    canvas.draw()

    # Convert the rendered buffer to a NumPy array
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))  # Shape (height, width, 3)
    return image

class LightningEncoderERM(L.LightningModule):
    def __init__(self, args, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.preds =  defaultdict(lambda: defaultdict(list))
        self.labels =  defaultdict(lambda: defaultdict(list))
        
    def forward(self, batch):

        x, y = batch

        bs, *_ = x[0].shape
        
        x = torch.cat([x[0] , x[1]], dim=0) # concatenate along batch dimension for efficiency
        features = self.encoder(x) 
        x_1 = features[:bs]
        x_2 = features[bs:]

        features = torch.cat([x_1, x_2], dim=-1)

        output = self.predictor(features)
        
        losses = []
        # decompose loss per task
        for i, fov in enumerate(self.args.fovs_tasks):
            losses.append(self.criterion(output[i], y[:,i]))
        
        return torch.stack(losses), [o.detach().cpu() for o in output]

    def training_step(self, batch, batch_idx):
        expected_output = batch[-1]
        expected_output = expected_output.cpu()
        losses, outputs = self(batch)
        loss = torch.mean(losses)
        metrics = {"train_loss": loss}
        for i, l in enumerate(losses):
            metrics[f"train_{self.args.fovs_tasks[i]}_loss"] = l.cpu().detach().item()
        
        for i, output in enumerate(outputs):
            fov = self.args.fovs_tasks[i]
            preds = output.argmax(dim=1)
            y = expected_output[:,i]
            correct = (preds == y).float().sum()
            total = y.numel()
            acc = correct/total
            metrics[f'train_{fov}_acc'] = acc
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        expected_output = batch[-1]
        expected_output = expected_output.cpu()
        losses, outputs = self(batch)
        loss = torch.mean(losses)
        split = ['val','test'][dataloader_idx]
        metrics = {f"{split}_loss": loss}
        for i, l in enumerate(losses):
            metrics[f"{split}_{self.args.fovs_tasks[i]}_loss"] = l.cpu().detach().item()
        for i, output in enumerate(outputs):
            fov = self.args.fovs_tasks[i]
            preds = output.argmax(dim=1)
            y = expected_output[:,i]
            correct = (preds == y).float().sum()
            total = y.numel()
            acc = correct/total
            metrics[f'{split}_{fov}_acc'] = acc
            
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        expected_output = batch[-1]
        expected_output = expected_output.cpu()
        losses, outputs = self(batch)
        loss = torch.mean(losses)
        split = ['id','shape','scale','orientation','x','y'][dataloader_idx]
        metrics = {f"{split}_loss": loss}
        for i, l in enumerate(losses):
            metrics[f"{split}_{self.args.fovs_tasks[i]}_loss"] = l.cpu().detach().item()
        for i, output in enumerate(outputs):
            fov = self.args.fovs_tasks[i]
            preds = output.argmax(dim=1)
            self.preds[split][fov].append(output)
            y = expected_output[:,i]
            self.labels[split][fov].append(y)
            correct = (preds == y).float().sum()
            total = y.numel()
            acc = correct/total
            metrics[f'{split}_{fov}_acc'] = acc
            
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def on_test_epoch_end(self):
        for split, data in self.preds.items():
            for fov, preds in data.items():
                self.preds[split][fov] = torch.cat(preds)
                labels = self.labels[split][fov]
                self.labels[split][fov] = torch.cat(labels).int()
        
        cms = dict()

        for split, data in self.preds.items():
            for fov, preds in data.items():
                labels = self.labels[split][fov]
                confusion_matrix = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=3, threshold=0.05)
                confusion_matrix(preds, labels)
                fig, _ = confusion_matrix.plot(labels=["same", "greater than", "lower than"])
                img = fig_to_img(fig)
                cms[f"{split}_cm_{fov}"] = torch.tensor(confusion_matrix.compute().detach().cpu().numpy().astype(int)).float()
                cms[f"{split}_cm_{fov}_img"] = img
                confusion_matrix = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=3, normalize="true", threshold=0.05)
                confusion_matrix(preds, labels)
                fig, _ = confusion_matrix.plot(labels=["same", "greater than", "lower than"])
                img = fig_to_img(fig)
                cms[f"{split}_cm_{fov}_perc"] = torch.tensor(confusion_matrix.compute().detach().cpu().numpy().astype(int)).float()
                cms[f"{split}_cm_{fov}_imgperc"] = img
                #self.log_dict({f"{split}_cm_{fov}": confusion_matrix_computed},on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

        # Clear all preds
        if not self.args.test:
            print(f"Saving to {self.args.experiment_id}_cm.pth")
            torch.save(cms, f"{self.args.experiment_id}_cm.pth")
        self.preds =  defaultdict(lambda: defaultdict(list)) # Reset validation preds
        self.labels =  defaultdict(lambda: defaultdict(list)) # Reset validation ground truth labels


    def configure_optimizers(self):
        param_groups = [
                {'params': self.encoder.parameters()},
                {'params': self.predictor.parameters()}
            ]
        return torch.optim.AdamW(param_groups, lr=self.args.lr)


class LightningTaskJEPA(L.LightningModule):
    def __init__(self, args, encoder, target_encoder):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.criterion = F.smooth_l1_loss
        self.args = args
        self.last_batch = -1 # Records id of last training batch seen
        
    def forward(self, batch):
        x, y = batch
        n_fovs = y.shape[-1]
        # create all pairs of input and target
        latents_zeros = torch.zeros_like(y)
        # Make image pairs (input, target) and assign correct latents
        x_input = x[0]
        x_target = x[1]
        # 4 cases 
        # Same rep:
        # x_0 with latent vs x_1
        # x_0 vs x_1 with -latent
        # Different rep
        # x_0 vs x_1
        # x_0 with latent vs x_1 with latent! 
        targets_1 = self.target_encoder(x_target, latents_zeros)
        output_1 = self.encoder(x_input, y)
        targets_2 = self.encoder(x_target, -y)
        output_2 = self.target_encoder(x_input, latents_zeros)
        return output_1, targets_1, output_2, targets_2

    def on_after_backward(self): # For applying ema
        # Momentum update of target encoder
        
        with torch.no_grad():
            m = next(args.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        
    def training_step(self, batch, batch_idx):
        self.last_batch = batch_idx
        output_1, targets_1, output_2, targets_2 = self(batch)
        same_loss, diff_loss = self.calculate_loss(output_1, targets_1, output_2, targets_2)
        loss = same_loss + diff_loss
        metrics = {'train_loss': loss, 'train_same_loss': same_loss, 'train_diff_loss': diff_loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def calculate_loss(self, o1, t1, o2, t2, eps=0.01):
        same_loss = self.criterion(o1, t1) + self.criterion(o2, t2)      # Should have same reps
        margin = torch.tensor(1.).to(o1.device) # send margin to correct device
        diff_loss = 1/(torch.min(margin, self.criterion(o1, t2)) + eps)     # Should have different reps
        
        return same_loss, diff_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = ['val','test'][dataloader_idx]            
        output_1, targets_1, output_2, targets_2 = self(batch)
        same_loss, diff_loss = self.calculate_loss(output_1, targets_1, output_2, targets_2)
        loss = same_loss + diff_loss
        metrics = {f'{split}_loss': loss, f'{split}_same_loss': same_loss, f'{split}_diff_loss': diff_loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)


    def configure_optimizers(self):
        param_groups = [
                {'params': self.encoder.parameters()}
            ]
        return torch.optim.AdamW(param_groups, lr=self.args.lr)

class LightningStudentRep(L.LightningModule):
    def __init__(self, args, encoder):
        super().__init__()
        self.encoder = encoder
        self.criterion = F.smooth_l1_loss
        self.args = args
        
    def forward(self, batch): # batch should be x(x0,x1), y (x0-x1) latent diff between 0 and 1
        x, reps_x, latent = batch
        # create all pairs of input and target
        latents_zeros = torch.zeros_like(latent)
        # Make image pairs (input, target) and assign correct latents
        output_1 = self.encoder(x[0], latent) # target = y[1]
        output_2 = self.encoder(x[1], -latent) # target = y[0]
        output_3 = self.encoder(x[0], latents_zeros) # target = y[0]
        output_4 = self.encoder(x[1], latents_zeros) # target = y[1]
        
        return output_1, output_2, output_3, output_4, reps_x[0], reps_x[1]

    # We try to replicate the learned representation as a start.
    # We learn to replicate the representation when latents are zero
    # and also learn to replicate it when latents are different to zero.
    def training_step(self, batch, batch_idx):
        x0_l, x1_l, x0, x1, x0_rep, x1_rep = self(batch)

        # Trying to copy
        # x0_l == x1
        # x1_l == x0

        latent_loss = self.criterion(x0_l, x1_rep) + self.criterion(x1_l, x0_rep)
        same_loss = self.criterion(x0, x0_rep) + self.criterion(x1, x1_rep)
        loss = latent_loss + same_loss
        metrics = {'train_loss': loss, 'train_same_loss': same_loss, 'train_latent_loss': latent_loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = ['val','test'][dataloader_idx]            
        x0_l, x1_l, x0, x1, x0_rep, x1_rep = self(batch)

        # Trying to copy
        # x0_l == x1
        # x1_l == x0

        latent_loss = self.criterion(x0_l, x1_rep) + self.criterion(x1_l, x0_rep)
        same_loss = self.criterion(x0, x0_rep) + self.criterion(x1, x1_rep)
        loss = latent_loss + same_loss
        metrics = {f'{split}_loss': loss, f'{split}_same_loss': same_loss, f'{split}_latent_loss': latent_loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)


    def configure_optimizers(self):
        param_groups = [
                {'params': self.encoder.parameters()}
            ]
        return torch.optim.AdamW(param_groups, lr=self.args.lr)

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
    
    output_path = f"results/{args.dataset}"
    csv_logger = CSVLogger(
        output_path,
        version = "v1",
        name=args.experiment_id
    )
    csv_logger.log_hyperparams(args)
    loggers.append(csv_logger)
    
    return loggers
    
def create_callbacks(args):
    callbacks = []
    save_top_k=-1
    dir_path = f"results/{args.dataset}/{args.experiment_id}"
    callbacks.append(ModelCheckpoint(dirpath=dir_path,
                                 filename="{epoch}",
                                 save_top_k=save_top_k,
                                 monitor="val_loss",
                                 every_n_epochs=5,
                                 save_last=True))
    return callbacks


def get_lightning_model(args):
    encoder, predictor = create_model(args)
    
    if args.train_method == "encoder_erm":
        model = LightningEncoderERM(args, encoder,predictor)
    
    elif args.train_method == "task_jepa":
        model = LightningTaskJEPA(args, encoder, predictor)

    elif args.train_method == "rep_train":
        model = LightningStudentRep(args, encoder)
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
    dls = get_dataloaders(args)
    loggers = create_loggers(args)
    args.experiment_name = get_exp_name(args)
    callbacks = create_callbacks(args)
    wandb.run.name = args.experiment_name

    model = get_lightning_model(args)
    
    ckpt_path = f"results/{args.experiment_id}/last.ckpt" if args.resume else None
    
    args = add_EMA_args(args)
    
    trainer = L.Trainer(accelerator="gpu",
                        devices=1,
                        max_epochs=args.num_epochs,
                        callbacks=callbacks,
                        logger=loggers)
    trainer.fit(model=model,
                train_dataloaders=dls['train'],
                val_dataloaders=[dls['val'],dls['test']],
                ckpt_path = ckpt_path)
    
    
    wandb.finish()