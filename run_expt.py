from datasets import get_dataloaders

from easydict import EasyDict as edict
from train import train
import wandb
import argparse

parser = argparse.ArgumentParser(description="Example of argparse usage")
# Add arguments
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--train_method', type=str, choices=['erm','task_jepa',"pair_erm","encoder_erm"], default="erm", help='Training Method')
parser.add_argument('--dataset', type=str, default="shapes3d", help='Dataset')
parser.add_argument('--seed', type=int, default=111, help='Seed')
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
args.encoder = {'pretrained_feats': False, 'output_dim': 384, 'arch': 'vit', 'frozen': False}
dls = get_dataloaders(args)
train(args, dls)