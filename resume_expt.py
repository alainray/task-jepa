from datasets import get_dataloaders

from easydict import EasyDict as edict
from train_lightning import train
import wandb
import argparse
import os
from utils import get_args

os.environ["TORCH_HOME"] = "~/storage/cache"
parser = argparse.ArgumentParser(description="Example of argparse usage")
# Add arguments
parser.add_argument('--resume_id', type=str, help='Resume Experiment Id')

# Parse the arguments
parsed_args = parser.parse_args()

args = get_args(parsed_args.resume_id,update_id=True)
args.resume_id = parsed_args.resume_id

train(args)