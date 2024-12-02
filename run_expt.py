from datasets import get_dataloaders

from easydict import EasyDict as edict
from train import train
import wandb
import argparse

parser = argparse.ArgumentParser(description="Example of argparse usage")
# Add arguments
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--train_method', type=str, choices=['erm','task_jepa',"pair_erm","encoder_erm"], default="erm", help='Training Method')
parser.add_argument('--frozen', action="store_true", help='Encoder is frozen or not')
parser.add_argument('--resume', action="store_true", help='Resume Experiment or not')
parser.add_argument('--experiment_id', type=str, default=None, help='Experiment id to resume run')
parser.add_argument('--pretrained_id', type=str, default=None, help='Experiment id associated with checkpoint')
parser.add_argument('--pretrained_epoch', type=int, default=50, help='Which epoch the checkpoint belongs to')
parser.add_argument('--arch', type=str, choices=["lvit","vit","vit_b_16","vit_b_32","vit_l_16","vit_l_32"], default="vit", help='Encoder Architecture')
parser.add_argument('--dataset', type=str, default="shapes3d", help='Dataset')
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
#parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
#parser.add_argument('--model', type=str, required=True, help='Model type to use (e.g., resnet, cnn)')

# Parse the arguments
parsed_args = parser.parse_args()

args = edict()

for k,v in vars(parsed_args).items():
    args[k] = v
    
#args.num_epochs = 2
#args.train_method = "erm"
args.from_pretrained = ""
args.test = False
METRICS_PER_METHOD = {"erm": ['acc'], "task_jepa": [], "ijepa": [], "task_jepa+erm": ['acc'], "pair_erm": ["acc"], "encoder_erm": ['acc']}
FOVS = {"shapes3d": {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}}
FOVS_PER_DATASET = {'shapes3d': ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]}
#FOVS_PER_DATASET = {'shapes3d': ["floor_hue", "wall_hue", "scale", "shape", "orientation"]}
args.best_model_criterion = "val_avg_acc"
args.metrics = METRICS_PER_METHOD[args.train_method]
args.dataset = "shapes3d"
args.fovs =  FOVS_PER_DATASET[args.dataset]
args.fovs_tasks = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
args.fovs_indices = {name: i for i, name in enumerate(args.fovs)}
# When 2, 0 == same, 1 = different, When 3 levels, 0 = same, 1 = greater than, 2 = lower than
args.fovs_levels = {"floor_hue": 2, "wall_hue": 2, "object_hue": 2, "scale": 3, "shape": 2, "orientation": 3}
args.fovs_ids = [args.fovs_indices[x] for x in args.fovs_tasks ]
args.n_fovs = FOVS[args.dataset]
args.task_to_label_index = {k: i for i, (k, v) in enumerate(FOVS[args.dataset].items())}
args.data_dir = "/mnt/nas2/GrimaRepo/araymond/3dshapes"
args.encoder = {'pretrained_feats': False,
                 'arch': args.arch,
                'frozen': args.frozen,
                'id': args.pretrained_id,
                'epoch': args.pretrained_epoch,}
dls = get_dataloaders(args)

# Saving metrics and checkpoints
args.save_weights = not args.test # only save checkpoints when not testing
args.save_metrics = not args.test
args.save_every = 10               # 

# optimization
args.warmup = 2.0/15.0*args.num_epochs



train(args, dls)