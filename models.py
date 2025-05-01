import lightning as L
from torchvision.models import VisionTransformer
from model_info import encoder_constructor,encoders, modulators,regressors,weights, model_output_dims
from torchvision.models.vision_transformer import Encoder
from typing import Optional, Callable, List, NamedTuple
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import get_args, update_dict

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU
# It assumes a pair of images concatenated along the sequence dimension

class LightningRegression(L.LightningModule):
    def __init__(self, args, encoder, modulator, regressor,**kwargs):
        super().__init__()
        self.encoder = encoder
        self.use_reps = encoder is None
        self.modulator = None
        self.regressor = regressor
        self.criterion = F.cosine_similarity
        self.args = args
    
    # We try to replicate the learned representation as a start.
    # We learn to replicate the representation when latents are zero
    # and also learn to replicate it when latents are different to zero.
    def training_step(self, batch, batch_idx):
        
        split = "train"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict({k: v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['train_loss'] # so lightning can train

    def get_metrics(self, data):
        
        metrics = dict()
        loss = 0
       
        reg_loss = (data['logits']-data['targets'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.mean(dim=0)                # Average over batch

        loss +=  reg_loss.mean()

        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'reg_{task}'] = reg_loss[i]
            
        metrics['loss'] = loss
        return metrics 

    def split_step(self, batch):    
        # Batch is simple
        imgs, gt_reps, latents = batch
        mid_reps = gt_reps if self.use_reps else self.encoder(imgs.float(), gt_reps)     # Image encoding
        mid_reps = torch.nn.functional.normalize(mid_reps, p=2.0, dim=1, eps=1e-12)
        logits = self.regressor(mid_reps)
        data = dict()
        data['logits'] = logits
        data['targets'] = latents
        return data

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = "val"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['val_loss']
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        split = "test"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['test_loss']

    def configure_optimizers(self):        
        params = []

        if hasattr(self, 'encoder') and self.encoder is not None:
            params += list(self.encoder.parameters())
        if hasattr(self, 'modulator') and self.modulator is not None:
            params += list(self.modulator.parameters())
        if hasattr(self, 'regressor') and self.regressor is not None:
            params += list(self.regressor.parameters())
        
        param_groups = [{'params': params}]

        return torch.optim.AdamW(param_groups, lr=self.args.lr)


class LightningTransformRegression(L.LightningModule):
    def __init__(self, args, encoder, modulator,**kwargs):
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
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict({k: v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['train_loss'] # so lightning can train


    def forward(self, x, rep_x):

        reps = rep_x if self.use_reps else self.encoder(x.float())     # Image encoding
        bs = x.shape[0]

        l = torch.zeros((bs, len(self.args.FOVS_PER_DATASET)), 
                            dtype=reps.dtype,
                            device=reps.device)
        return self.modulator(reps, l)

    def get_metrics(self, data):
        
        metrics = dict()
        loss = 0
        reg_loss = (data['logits']-data['targets'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.sum(dim=1)                # Average over batch
        loss +=  reg_loss.mean() # TODO: We need to average over tasks.
        n_attrs = len(self.args.FOVS_PER_DATASET)
        dtype = reg_loss.dtype
        device = reg_loss.device
        sum_per_group = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                            data['tasks'],
                                                                            reg_loss,
                                                                            reduce="sum")


        counts = torch.zeros(n_attrs, dtype=data['tasks'].dtype, device=device).scatter_reduce(0, data['tasks'], torch.ones_like(data['tasks']).cuda(), reduce="sum")
        mean_per_group = sum_per_group/counts
        
        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'reg_{task}'] = mean_per_group[i]

        metrics['loss'] = loss
        return metrics 

    def split_step(self, batch):    
        # Batch is simple
        src_img, src_rep, imgs, gt_reps, deltas, src_latents, latents = batch
        bs, n_classes, _ = latents.shape
        #mid_reps = gt_reps if self.use_reps else self.encoder(imgs.float(), gt_reps)     # Image encoding
        src_rep = src_rep.unsqueeze(1).repeat((1,n_classes,1))
        reps = self.modulator(src_rep, deltas)                        # predicted reps given latents
        data = dict()
        data['logits'] = reps.view(bs*n_classes, -1)
        data['targets'] = gt_reps.view(bs*n_classes, -1)
        tasks = deltas.abs().argmax(dim=-1)
        data['tasks'] = tasks.view(-1)
        
        return data

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = "val"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['val_loss']
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        split = "test"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['test_loss']

    def configure_optimizers(self):        
        params = []

        if hasattr(self, 'encoder') and self.encoder is not None:
            params += list(self.encoder.parameters())
        if hasattr(self, 'modulator') and self.modulator is not None:
            params += list(self.modulator.parameters())
        if hasattr(self, 'regressor') and self.regressor is not None:
            params += list(self.regressor.parameters())
        
        param_groups = [{'params': params}]

        return torch.optim.AdamW(param_groups, lr=self.args.lr)



class LightningTransformPlusRegression(L.LightningModule):
    def __init__(self, args, encoder, modulator, regressor,**kwargs):
        super().__init__()
        self.encoder = encoder
        self.use_reps = encoder is None
        self.modulator = modulator
        self.regressor = regressor
        self.criterion = F.cosine_similarity
        self.args = args
    
    # We try to replicate the learned representation as a start.
    # We learn to replicate the representation when latents are zero
    # and also learn to replicate it when latents are different to zero.
    def training_step(self, batch, batch_idx):
        
        split = "train"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict({k: v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['train_loss'] # so lightning can train


    def forward(self, x, rep_x):

        reps = rep_x if self.use_reps else self.encoder(x.float())     # Image encoding
        bs = x.shape[0]

        l = torch.zeros((bs, len(self.args.FOVS_PER_DATASET)), 
                            dtype=reps.dtype,
                            device=reps.device)
        return self.modulator(reps, l)

    def get_regression_loss(self, data):
        
        metrics = dict()
        loss = 0
        
        reg_loss = (data['reg_preds']-data['reg_targets'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.mean(dim=0)                # Average over batch

        loss +=  reg_loss.mean()

        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'lat_reg_{task}'] = reg_loss[i]
            
        metrics['lat_loss'] = loss

        return metrics

    def get_metrics(self, data):
        
        metrics = dict()
        loss = 0
        reg_loss = (data['logits']-data['targets'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.sum(dim=1)                # Average over batch
        loss +=  reg_loss.mean()
        n_attrs = len(self.args.FOVS_PER_DATASET)
        dtype = reg_loss.dtype
        device = reg_loss.device
        sum_per_group = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                            data['tasks'],
                                                                            reg_loss,
                                                                            reduce="sum")


        counts = torch.zeros(n_attrs, dtype=data['tasks'].dtype, device=device).scatter_reduce(0, data['tasks'], torch.ones_like(data['tasks']).cuda(), reduce="sum")
        mean_per_group = sum_per_group/counts
        
        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'reg_{task}'] = mean_per_group[i]

        metrics['loss'] = loss

        m = self.get_regression_loss(data)
        for k, v in m.items(): # copy regression metrics to metrics dict
            metrics[k] = v
        metrics['loss'] += self.args.lambda_latent_loss*m['lat_loss']
        return metrics 

    def split_step(self, batch):    
        # Batch is simple
        src_img, src_rep, imgs, gt_reps, latents = batch
        bs, n_classes, _ = latents.shape
        #mid_reps = gt_reps if self.use_reps else self.encoder(imgs.float(), gt_reps)     # Image encoding
        src_rep = src_rep.unsqueeze(1).repeat((1,n_classes,1))
        reps = self.modulator(src_rep, latents)                        # predicted reps given latents
        data = dict()
        data['logits'] = reps.view(bs*n_classes, -1)
        data['targets'] = gt_reps.view(bs*n_classes, -1)
        tasks = latents.abs().argmax(dim=-1)
        data['tasks'] = tasks.view(-1)
        reg_preds = self.regressor(reps)
        data['reg_preds'] = reg_preds.view(bs*n_classes,-1)
        data['reg_targets'] = latents.view(bs*n_classes,-1)
        
        return data

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = "val"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['val_loss']
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        split = "test"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['test_loss']

    def configure_optimizers(self):        
        params = []

        if hasattr(self, 'encoder') and self.encoder is not None:
            params += list(self.encoder.parameters())
        if hasattr(self, 'modulator') and self.modulator is not None:
            params += list(self.modulator.parameters())
        if hasattr(self, 'regressor') and self.regressor is not None:
            params += list(self.regressor.parameters())
        
        param_groups = [{'params': params}]

        return torch.optim.AdamW(param_groups, lr=self.args.lr)

# Possible Architectures

# vit_b_16 224 x 224
# vit_b_32 224 x 224
# vit_l_16 224 x 224
# vit_l_32 224 x 224
#

# A class that can do regression or classification depending on args.train_method
class LightningVersatile(L.LightningModule):
    def __init__(self, args, encoder, modulator, regressor, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.use_reps = encoder is None
        self.regressor = regressor
        self.modulator = modulator
        self.criterion = F.cosine_similarity
        self.args = args
        self.loss_methods = {'class': self.class_loss,
                             "same": self.same_loss,
                             'non_mod_reg': self.regression_loss,
                             'mod_reg': self.regression_loss,
                             "orth": self.orth_loss
                             }
    # All losses should take in data and return a dict with all the metrics
    # If we want to add it to the training loss, there should be a key named "loss"
    # in this dict.

    def orth_loss(self, data, prefix=""):
        metrics = dict()
        params = self.modulator.proj_latent_dirs
        n_params = nn.functional.normalize(params, p=2, eps=1e-12, dim=-1)
        ort_loss = torch.matmul(n_params, n_params.T).abs()
        mask = torch.ones_like(ort_loss)
        indices = torch.arange(ort_loss.size(-1))
        mask[indices, indices] = 0
        # Apply the mask
        ort_loss_no_diag = ort_loss * mask
        ort_loss = ort_loss.mean()
        metrics['orth_loss'] = ort_loss
        metrics['loss'] = ort_loss
        return metrics

    def class_loss(self, data, prefix=""):
        # requires the following keys in data:
        # ("logits", "class_tgt", "tasks")
        # "logits" is a (-1, n_classes) vector
        # "class_tgt" is a (-1, 1) vector
        # "tasks" is a a (-1, 1) vector
        metrics = dict()

        # Get classification loss
        class_loss = F.cross_entropy(data['class_logits'], data['class_tgt'], reduction="mean")
        metrics['class_loss'] = class_loss.clone()
        # Get total accuracy
        preds = data['class_logits'].argmax(dim=-1).view(-1)
        correct = (preds == data['class_tgt']).view(-1).float()
        accuracy = correct.sum()/correct.numel()
        metrics['class_acc'] = accuracy
        
        # Get accuracy per latent attribute!
        dtype=correct.dtype
        device=correct.device
        tasks = data['class_tasks'].view(-1)
        n_attrs = len(self.args.FOVS_PER_DATASET)
    
        sum_per_group = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                            tasks,
                                                                            correct,
                                                                            reduce="sum")

        counts = torch.zeros(n_attrs, dtype=tasks.dtype, device=device).scatter_reduce(0, tasks, torch.ones_like(tasks).cuda(), reduce="sum")
        mean_per_group = sum_per_group/counts

        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'class_{task}'] = mean_per_group[i]
            
        metrics['loss'] = class_loss
        
        return metrics

    def encode(self, imgs, reps_imgs):
        enc_reps = reps_imgs if self.use_reps else self.encoder(imgs.float(), reps_imgs)     # Image encoding
        return enc_reps

    def modulate(self, reps, deltas):
        new_reps = self.modulator(reps, deltas)
        if "unnormalized_modulator" not in self.args or not self.args.unnormalized_modulator:
            return torch.nn.functional.normalize(new_reps, p=2.0, dim=-1, eps=1e-12)
        else:
            return new_reps

    def regression_loss(self, data, prefix = ""):
        
        metrics = dict()
        loss = 0
        reg_loss = (data[f'{prefix}_preds']-data[f'{prefix}_tgts'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.mean(dim=0)                # Average over batch

        loss +=  reg_loss.mean()

        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'{prefix}_reg_{task}'] = reg_loss[i]
            
        metrics[f'{prefix}_loss'] = loss.clone()
        metrics['loss'] = loss

        return metrics

    def same_loss(self, data, prefix=""):
        metrics = dict()
        dot_prods = (data['same_logits']*data['same_tgts']).sum(dim=-1).mean()
        loss = 1 - dot_prods
        metrics['same_loss'] = loss.clone()
        metrics['loss'] = loss
        
        return metrics
    
    def predict_regression(self, batch):
        imgs, reps, latents = batch
        reps = self.encode(imgs, reps)
        zero_deltas = torch.zeros_like(latents).float()
        if self.modulator is not None:
            reps = self.modulate(reps, zero_deltas)
        preds = self.regressor(reps)
        data = dict()
        data['logits'] = preds
        data['targets'] = latents
        return data

    def get_modulated_reps(self, batch):
        # batch should be 
        imgs, reps, deltas = batch
        reps = self.encode(imgs, reps)
        reps = self.modulate(reps, deltas)
        return reps

    def split_step(self, batch):    
        data = dict()
        src_img, src_rep, imgs, gt_reps, deltas, src_latents, latents = batch
        zero_latents = torch.zeros_like(latents)
        bs, n_classes, c, h, w = imgs.shape
        imgs = imgs.view(bs*n_classes, c, h, w) if self.args.encoder.arch == "cnn" else imgs
        img_reps = self.encode(imgs.float(), gt_reps)     # Image encoding
        img_reps = img_reps.view(bs, n_classes, -1) if self.args.encoder.arch == "cnn" else img_reps

        # Get encoder reps! # input images --> output reps!
        src_rep = self.encode(src_img.float(), src_rep)     # Image encoding        
        src_rep = src_rep.unsqueeze(1).repeat((1,n_classes,1))
        
        # Pass encoder reps through modulator --> input: images and deltas --> output modulated reps and gt reps
        if self.args.modulator is not None:
            mod_reps = self.modulate(src_rep, deltas)                        # predicted reps given latents
            non_mod_reps = self.modulate(src_rep[:,0].view(bs,-1),
                                         zero_latents[:,0].view(bs,-1)) # non modulated source reps
            tgt_reps = self.modulate(img_reps, zero_latents)               # non modulated reps for target
        
        # Classification Task
        if "class" in self.args.losses:
            # Get classification logits!
            data['class_logits'] = torch.matmul(mod_reps, tgt_reps.transpose(1,2)).view(-1, n_classes) # bs x 10 x 10 --> 10bs x 10
            data['class_tgt'] = torch.tensor(bs*list(range(n_classes))).view(-1).to(data['class_logits'].device)
            data['class_tasks'] = deltas.abs().argmax(dim=-1)

        if "same" in self.args.losses:
            data['same_logits'] = torch.nn.functional.normalize(mod_reps, p=2.0, dim=-1, eps=1e-12)
            data['same_tgts'] = torch.nn.functional.normalize(tgt_reps , p=2.0, dim=-1, eps=1e-12)
    
        # Regression Task!

        # Add manipulated reps and non-manipulated reps
        # 2 possible regressions!
        #    a. Modulated regression
        #    b. Non Modulated Regression: (baseline)

        # loss: mod regression: encoder --> modulator (with delta != 0) --> regression
        if "mod_reg" in self.args.losses:
            data['mod_reg_preds'] = self.regressor(mod_reps.view(bs*n_classes,-1))
            data['mod_reg_tgts'] = latents.view(bs*n_classes,-1)
        
        # loss: non modulated regression
        if "non_mod_reg" in self.args.losses:
            if self.modulator is None:         # loss: non modulated regression: encoder --> regression
                non_mod_reps = torch.cat([src_rep[:,0].view(bs,-1), img_reps.view(bs*n_classes,-1)], dim=0)
            else:
                # loss: non modulated regression: encoder --> modulator (with delta = 0) --> regression
                non_mod_reps = torch.cat([non_mod_reps, tgt_reps.view(bs*n_classes,-1)], dim=0)
        
            non_mod_latents = torch.cat([src_latents.view(bs,-1), latents.view(bs*n_classes,-1)], dim=0)
            data['non_mod_reg_preds'] = self.regressor(non_mod_reps)
            data['non_mod_reg_tgts'] = non_mod_latents
            
        return data

    # Register all metrics from all losses
    def step(self, batch):
                
        data = self.split_step(batch)
        metrics = {}
        for loss in self.args.losses:
            m = self.loss_methods[loss](data, prefix=loss)
            metrics = update_dict(metrics, m)
        return metrics # so lightning can train

    def log_metrics_split(self, metrics, split):
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict({k: v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics # so lightning can train

    def training_step(self, batch, batch_idx):
        split = "train"
        metrics = self.step(batch)
        metrics = self.log_metrics_split(metrics, split)
        return metrics[f'{split}_loss']
      
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = "val"
        metrics = self.step(batch)
        metrics = self.log_metrics_split(metrics, split)
        return metrics[f'{split}_loss']
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        split = "test"
        metrics = self.step(batch)
        metrics = self.log_metrics_split(metrics, split)
        return metrics[f'{split}_loss']
    def configure_optimizers(self):        
        params = []

        if hasattr(self, 'encoder') and self.encoder is not None:
            params += list(self.encoder.parameters())
        if hasattr(self, 'modulator') and self.modulator is not None:
            params += list(self.modulator.parameters())
        if hasattr(self, 'regressor') and self.regressor is not None:
            params += list(self.regressor.parameters())
        
        param_groups = [{'params': params}]

        return torch.optim.AdamW(param_groups, lr=self.args.lr)

def get_lightning_model(args, encoder=None, modulator=None, ckpt_path=None):

    # train method --> which losses they use, which modules (encoder, modulator, regressor) and the type of batch they take
    lightning_model_builder = {
                                'rep_train': LightningVersatile, # encoder + modulator + class_loss 
                                'regression': LightningRegression, # encoder + regressor + non_mod_reg loss
                                'transform': LightningTransformRegression, # CAN'T DO YET
                                'transform_plus': LightningTransformPlusRegression, # CAN'T DO YET
                                'rep_train_plus': LightningVersatile, # encoder + modulator + regressor + class_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_same': LightningVersatile, # encoder + modulator + regressor + same_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_plus_res': LightningVersatile, # encoder + modulator (residual) + regressor + class_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_plus_trans': LightningVersatile, # encoder + modulator (transformer) + regressor + class_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_plus_film': LightningVersatile, # encoder + modulator (film) + regressor + class_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_same_res': LightningVersatile, # encoder + modulator (residual) + regressor + same_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_same_trans': LightningVersatile, # encoder + modulator (transformer) + regressor + same_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_same_film': LightningVersatile, # encoder + modulator (film) + regressor + same_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_same_linop': LightningVersatile, # encoder + modulator (linear_operator) + regressor + same_loss + non_mod_reg_loss + mod_reg_loss
                                'rep_train_same_latdir': LightningVersatile, # encoder + modulator (latent directions) + regressor + same_loss + non_mod_reg_loss + mod_reg_loss + orthogonal loss
                                'mod_regression': LightningVersatile ,    # encoder + modulator + regressor + non_mod_reg_loss + mod_reg_loss
                                'mod_regression_trans': LightningVersatile ,# encoder + modulator (transformer) + regressor + non_mod_reg_loss + mod_reg_loss
                                'mod_regression_film': LightningVersatile ,# encoder + modulator (film) + regressor + non_mod_reg_loss + mod_reg_loss
                                'mod_regression_linop': LightningVersatile ,# encoder + modulator (linear operator) + regressor + non_mod_reg_loss + mod_reg_loss
                                'mod_regression_latdir': LightningVersatile ,# encoder + modulator (latent direction) + regressor + non_mod_reg_loss + mod_reg_loss
                                'non_mod_regression': LightningVersatile # encoder + modulator(transformer) + regressor + non_mod_reg_loss + mod_reg_loss
                                
                              }
    
    if ckpt_path is None:
        encoder, modulator, regressor = create_model(args)
        model = lightning_model_builder[args.train_method](args, encoder=encoder, 
                                                                modulator=modulator,
                                                                regressor=regressor)
    else:
        # load from checkpoint
        print(f"Loading Pretrained Encoder from {ckpt_path}")
        model = lightning_model_builder[args.train_method].load_from_checkpoint(
                                                    checkpoint_path=ckpt_path,
                                                    args=args,
                                                    encoder=encoder,
                                                    modulator=modulator,
                                                    regressor=regressor
                                                    )
        
    return model

def create_model(args): 
    global weights
    # baseline architectures
    # Define encoder
    # Define modulator
    # Define regressor
    # Freeze/unfreeze weights
    regressor = None
    if "pretrained_encoder" in args and args.pretrained_encoder is not None:
        print(f"Creating model from pretrained encoder in {args.pretrained_encoder}")
        encoder_args = get_args(args.pretrained_encoder) # id of experiment to get the model from/only used from a rep_train experiment
        encoder, modulator = create_model(encoder_args)
        ckpt_path = f"results/{encoder_args.dataset}/{args.pretrained_encoder}/last.ckpt"
        encoder = get_lightning_model(
                                        encoder_args,
                                        encoder=encoder,
                                        modulator=modulator,
                                        ckpt_path=ckpt_path
                                    )
        input_dims = encoder_args.modulator.hidden_dim if encoder_args.train_method == "rep_train" else model_output_dims[encoder_args.pretrained_reps]# output of pretrained_encoder
        print(f"Input Dims to modulator will be: {input_dims}")
    else:
        model_weights = weights[args.encoder.arch] if args.encoder.pretrained else None 
        encoder = encoders[args.encoder.arch](args, weights=model_weights) if args.encoder.arch != "none" else None
        input_dims = model_output_dims[args.encoder.arch] if args.encoder.arch != "none" else  model_output_dims[args.pretrained_reps]


    modulator = modulators[args.train_method](input_dim=input_dims,
                                              hidden_dim= input_dims if args.train_method in ["transform","transform_plus"] else args.modulator.hidden_dim,
                                              latent_dim = len(args.FOVS_PER_DATASET)
                                              ) if modulators[args.train_method] is not None else None

    if args.train_method != "rep_train":
        regressor = regressors[args.train_method](input_dim=input_dims if args.train_method in ["transform_plus", "regression"] else args.modulator.hidden_dim,
                                              hidden_dim=args.modulator.hidden_dim,
                                              latent_dim = len(args.FOVS_PER_DATASET))
    # Freeze/unfreeze parameters
    if args.encoder.arch != "none":
        for p in encoder.parameters():
            p.requires_grad = not args.encoder['frozen']

    return encoder, modulator, regressor
