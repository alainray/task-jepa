import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from datasets import get_dataloaders

from easydict import EasyDict as edict
from train_lightning import train
import wandb
import argparse
import os

os.environ["TORCH_HOME"] = "~/storage/cache"
parser = argparse.ArgumentParser(description="Example of argparse usage")
# Add arguments
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--num_steps', type=int, default=400000, help='Number of iterations to train')
parser.add_argument('--dataset', type=str, default="shapes3d", help='Dataset')
parser.add_argument('--train_method', type=str, choices=['erm','task_jepa',"pair_erm","encoder_erm","rep_train"], default="erm", help='Training Method')
parser.add_argument('--pretrain_method', type=str, default=None, help='PreTraining Method')
parser.add_argument('--train_bs', type=int, default=400000, help='Batch size for training')
parser.add_argument('--frozen', action="store_true", help='Encoder is frozen or not')
parser.add_argument('--resume_id', type=str, default=None, help='Resume Experiment Id')
parser.add_argument('--losses', type=str, default="all", help='Which losses to use when rep training') # same/latent/all
parser.add_argument('--experiment_id', type=str, default=None, help='Experiment id to resume run')
parser.add_argument('--pretrained_id', type=str, default=None, help='Experiment id associated with checkpoint')
parser.add_argument('--pretrained_epoch', type=int, default=50, help='Which epoch the checkpoint belongs to')
parser.add_argument('--pretrained_model', action="store_true", help='Resume Experiment or not')
parser.add_argument('--pretrained_reps', type=str, default=None, help='Target for rep_train method')
parser.add_argument('--arch', type=str, default="vit", help='Encoder Architecture')
parser.add_argument('--mod_arch', type=str, default="mlp", help='Encoder Architecture')
parser.add_argument('--enc_dims', type=int, default=16, help='Number of output dims of encoder')
parser.add_argument('--mod_dims', type=int, default=16, help='Number of output dims of modulator')
parser.add_argument('--seed', type=int, default=111, help='Seed')
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for Data Loaders")
parser.add_argument("--ema_start", type=float, default=0.996, help="Starting factor for Exponential Moving Average")
parser.add_argument("--lr", type=float, default=0.001, help="Reference Learning Rate")
parser.add_argument("--start_lr", type=float, default=0.0002, help="Starting learning rate for scheduler")
parser.add_argument("--final_lr", type=float, default=1e-06, help="Final learning rate for scheduler")
parser.add_argument("--iters_per_ema", type=float, default=1.0, help="How many iterations before update of EMA")
parser.add_argument("--wd", type=float, default=0.04, help="Reference Weight Decay")
parser.add_argument("--final_wd", type=float, default=0.4, help="Final Weight Decay for scheduler")
parser.add_argument("--ipe_scale", type=float, default=1.0, help="Scale factor for iterations per epoch")


# Parse the arguments
parsed_args = parser.parse_args()

args = edict()

for k,v in vars(parsed_args).items():
    args[k] = v

args.sub_dataset = int(args.dataset.split("_")[1]) if "idsprites" in args.dataset else args.dataset.split("_")[1]
args.dataset = args.dataset.split("_")[0]
args.test = False
args.encoder = {
    'pretrained': args.pretrained_model,
   # 'output_dim': 384, # vit_b_16: 784, vit_b_32: 784 # vit: 384.
    'id': args.pretrained_id,
    "enc_dims": args.enc_dims,
    'epoch': args.pretrained_epoch,
    'arch': args.arch,
    'frozen': args.frozen,
    'pretrain_method': args.pretrain_method}

args.modulator = {'arch': args.mod_arch, 'hidden_dim': args.mod_dims}
args.pretrained_arch = args.encoder['arch'] if args.encoder['pretrained'] and args.encoder['frozen'] else None
args.data_dir = "idsprites" if args.dataset == "idsprites" else f"/mnt/nas2/GrimaRepo/araymond"  
args.pretrained_feats = False
args.experiment_id = None 
args.save_weights = not args.test
args.save_metrics = not args.test
args.save_every = 10

METRICS_PER_METHOD = {
                      "erm": ['acc'],
                      "task_jepa": ['acc'], 
                      "ijepa": ['acc'],
                      "task_jepa+erm": ['acc'],
                      "pair_erm": ["acc"],
                      "encoder_erm": ['acc'],
                        "rep_train": ["acc"]
                     }

FOVS = {"3dshapes": {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15},
        "idsprites": {'shape': 10, 'scale': 10, 'orientation': 10, 
                          'x': 8, 'y': 4}
       }
FOVS_PER_DATASET = {'3dshapes': ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"],
                   'idsprites': ["shape","scale","orientation","x","y"]
                   }
#FOVS_PER_DATASET = {'shapes3d': ["floor_hue", "wall_hue", "scale", "shape", "orientation"]}

args.metrics = METRICS_PER_METHOD[args.train_method]
args.fovs =  FOVS_PER_DATASET[args.dataset]
# Defines which tasks to optimize for when training under EncoderERM
args.fovs_tasks = args.fovs # ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
args.fovs_indices = {name: i for i, name in enumerate(args.fovs)}
# When 2, 0 == same, 1 = different, When 3 levels, 0 = same, 1 = greater than, 2 = lower than
args.fovs_levels = {
                    '3dshapes': {"floor_hue": 2, "wall_hue": 2, "object_hue": 2, "scale": 3, "shape": 2, "orientation": 3},
                    'idsprites': {"shape": 3, "scale": 3, "orientation": 3, "x": 3, "y": 3}
                   }
args.fovs_ids = [args.fovs_indices[x] for x in args.fovs_tasks ]
args.n_fovs = FOVS[args.dataset]
args.task_to_label_index = {k: i for i, k in enumerate(args.fovs_tasks)}

# optimization
args.warmup = 2.0/15.0*args.num_epochs

#dls = get_dataloaders(args)

# Saving metrics and checkpoints
args.save_every = 10               # 

# optimization
args.warmup = 2.0/15.0*args.num_epochs

def print_args(args):
    print("\n" + "="*40)
    print("           Training Arguments")
    print("="*40)
    for arg, val in vars(args).items():
        print(f"{arg:>20}: {val}")
    print("="*40 + "\n")
    
def main():
    print("Running experiment with args: ", flush=True)
    print_args(args)
    train(args)

if __name__ == "__main__":
    main()