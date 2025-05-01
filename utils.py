import torch
import numpy as np
from optim import CosineWDSchedule, WarmupCosineSchedule 
from torch.optim import AdamW, SGD
import os
import pandas as pd
import wandb
from easydict import EasyDict as edict


def update_dict(base_dict, new_dict):
    for key, value in new_dict.items():
        if key in base_dict:
            base_dict[key] += value
        else:
            base_dict[key] = value
    return base_dict

def get_args(run_id, update_id=False):
    # Initialize the W&B API
    api = wandb.Api()
    # Specify the project and run ID
    project_name = "task_jepa"
    # Fetch the run
    run = api.run(f"{wandb.api.default_entity}/{project_name}/{run_id}")
    # Get the hyperparameters
    args = edict(run.config)
    if update_id:
        args.experiment_id = run_id
    return args


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, value, n=1):
        """Updates the average meter with new value.

        Args:
            value (float): New value to add.
            n (int): Number of occurrences of the new value (default: 1).
        """
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns a string representation of the average meter."""
        return f'Average: {self.avg:.4f} (Count: {self.count})'

def get_exp_name(args):
    if "encoder" in args:
        encoder = args.encoder.arch + "_"
    
    experiment_id = ""
    
    if "experiment_id" in args:
        if args.experiment_id is not None:
            experiment_id = "_" + args.experiment_id

    return f"{args.dataset}_{args.train_method}_{encoder}{args.seed}{experiment_id}"

def setup_meters(args):
    
    metrics =  dict()
    
    if args.train_method in ["erm","pair_erm", "encoder_erm"]:
        
        args.metrics.append("loss")
       
        for fov in args.fovs_tasks:
            metrics[fov] = dict()
            metrics[fov] = {metric: AverageMeter() for metric in args.metrics}
    else:
      
        metrics = {
         "rep_loss": AverageMeter(),
         "rep_var_enc": AverageMeter(),
         "rep_var_tgt": AverageMeter()
         }
        
    return metrics

def update_meters(args, loss, output, expected_output, all_meters):
    if args.train_method in ["erm","pair_erm", "encoder_erm"]:
        #print(all_meters, loss)
        for i, (fov, meters) in enumerate(all_meters.items()):
            for metric, meter in meters.items():
                if metric == "loss":
                    val = loss[i]
                elif metric == "acc":
                    idx = args.task_to_label_index[fov]
                    preds = output[i].argmax(dim=1)
                    y = expected_output[:,idx]
                    correct = (preds == y).float().sum()
                    total = y.numel()
                    acc = correct/total
                    val = acc 
                all_meters[fov][metric].update(val) # update task metric
    else:
        all_meters["rep_loss"].update(loss)
        rep_variance1 = torch.var(output[0], dim=0).mean()
        rep_variance2 = torch.var(output[1], dim=0).mean()
        all_meters["rep_var_enc"].update(rep_variance1)
        all_meters["rep_var_tgt"].update(rep_variance2)
    
    return all_meters

def get_best_model(best_model, current_model, best_metrics, current_metrics, method="val_avg_acc"):
    # parse method
    split, task, metric = tuple(method.split("_"))
    if task == "avg":
        # get avg of all metrics
        mean = 0.0
        for task_metric, v in current_metrics[split].items():
            if isinstance(v, dict):
                for met, value in v.items():
                    if metric == met:
                        mean +=value
        mean/= len(current_metrics[split]) - 1
        new_metric = mean
        mean = 0.0
        for task_metric, v in best_metrics[split].items():
            if isinstance(v, dict):
                for met, value in v.items():
                    if metric == met:
                        mean +=value
        mean/= len(best_metrics[split]) - 1
        old_metric = mean 
    else:
        new_metric = current_metrics[split][f"{task}_{metric}"] 
        old_metric = best_metrics[split][f"{task}_{metric}"] 
    if new_metric > old_metric:
        return current_model, current_metrics
    else:
        return best_model, best_metrics
        
def format_metrics_wandb(metrics, split=""):
    full_metrics = dict()
    for k, v in metrics.items():
        if not isinstance(v, dict):
            v = {"repr_pred": v}
        for metric, value in v.items():
            if split != "":
                name = f"{split}_{k}_{metric}"
            else:
                name = f"{k}_{metric}"
            full_metrics[name] = value
    return full_metrics



def pprint_metrics(meter_dict):

    """Pretty prints a dictionary of AverageMeter objects."""
    for obj, metrics in meter_dict.items():
        print(f"{obj.capitalize()}:")

        if isinstance(metrics, dict):
            for metric, meter in metrics.items():
                # Assuming the AverageMeter has a method that returns average and count
                print(f"  - {metric.capitalize()}: Avg = {meter.avg:.4f}")
        else:
            print(f"  - Loss: Avg = {metrics.avg:.4f}")
        print()  # Add an empty line for better readability


def init_opt(
    args,
    models
):
    if args.train_method in ['task_jepa', 'ijepa']:
        encoder, target_encoder = models
        param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in target_encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in target_encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(args.warmup*args.ipe),
            start_lr=args.start_lr,
            ref_lr=args.lr,
            final_lr=args.final_lr,
            T_max=int(args.ipe_scale*args.num_epochs*args.ipe))
        wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=args.wd,
            final_wd=args.final_wd,
            T_max=int(args.ipe_scale*args.num_epochs*args.ipe))

    else:
        if args.train_method == "encoder_erm":
            param_groups = [
                {'params': models[0].parameters()},
                {'params': models[1].parameters()}
            ]
        else:
            param_groups = [
                {'params': models[0].parameters()}
            ]
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.wd)
        #optimizer = SGD(param_groups, lr=args.lr, weight_decay=args.wd)
        scheduler = None
        wd_scheduler = None

    return optimizer, scheduler, wd_scheduler

def save_model(args, models):
    results = {}
    for i, m in enumerate(models):
        results[f'model_{i}'] = m.state_dict()
    
    # Construct the directory path
    save_dir = f"models/{args.experiment_id}"
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the models
    save_path = f"{save_dir}/{args.epoch}.pth"
    torch.save(results, save_path)
    print(f"Model saved to {save_path}")

def save_metrics(args, all_metrics):
    # Convert list of dictionaries to a DataFrame
    
    # Construct the directory path
    save_dir = f"results/{args.experiment_id}"
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    for k, v in all_metrics.items():
        df = pd.DataFrame(v)
        df.to_csv(f"{save_dir}/{k}_{args.epoch}.csv")


def create_checkpoint(args, model, metrics):

    ckpt = dict()
    # Construct the directory path
    save_dir = f"results/{args.experiment_id}"
    ckpt_path = save_dir + "/last_checkpoint.pth"
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    n, w = next(iter(model[1].named_parameters()))
    print("Classifier 1", n, w)
    models = {f'model_{i}': m.state_dict() for i, m in enumerate(model) if m is not None}
    print("classifier", models["model_1"][n])
    ckpt['optimizer'] = args.optimizer.state_dict() if args.optimizer is not None else None
    print(f"Saved learning rate: {args.optimizer.param_groups[0]['lr']}")
    ckpt['scheduler'] = args.scheduler.state_dict() if args.scheduler is not None else None
    ckpt['wd_scheduler'] = args.wd_scheduler.state_dict() if args.wd_scheduler is not None else None
    ckpt['model'] = models
    ckpt['metrics'] = metrics
    ckpt['epoch'] = args.epoch
    ckpt['random_state'] = get_random_state()
    ckpt['momentum_scheduler'] = args.momentum_scheduler if "momentum_scheduler" in args else None
    temp_path = save_dir + "/temp.pth"
    torch.save(ckpt, temp_path)
    os.rename(temp_path, ckpt_path)

def get_random_state():
    checkpoint = {
        #'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    }
    return checkpoint

def restore_random_state(random_state):
    torch.set_rng_state(random_state['torch'])
    torch.cuda.set_rng_state_all(random_state['torch_cuda'])  # For all CUDA devices
    np.random.set_state(random_state['numpy'])
    #random.setstate(random_state['python'])

def resume_last_checkpoint(args, model):

    
    path_to_checkpoint = f"results/{args.experiment_id}/last_checkpoint.pth"
    ckpt = torch.load(path_to_checkpoint)
    
    args.epoch = ckpt['epoch']

    print("Loaded Encoder weights")
    model[0].load_state_dict(ckpt['model']['model_0'])
    
    if "model_1" in ckpt['model']:
        print("Loaded predictor weights")
        print(ckpt['model']['model_1']['classifiers.0.weight'])
        model[1].load_state_dict(ckpt['model']['model_1'])
    
    optimizer, scheduler, wd_scheduler = init_opt(args, model)
    if args.optimizer is not None:
        args.optimizer = optimizer
        args.optimizer.load_state_dict(ckpt['optimizer'])

    if args.scheduler is not None:
        args.scheduler = scheduler
        args.scheduler.load_state_dict(ckpt['scheduler'])
    if args.wd_scheduler is not None:
        args.wd_scheduler = wd_scheduler
        args.wd_scheduler.load_state_dict(ckpt['wd_scheduler'])
    if "momentum_scheduler" in ckpt:
        args.momentum_scheduler = ckpt['momentum_scheduler']
    restore_random_state(ckpt['random_state'])
    print(f"Restored learning rate: {args.optimizer.param_groups[0]['lr']}")
    
    return args, model, ckpt['metrics']