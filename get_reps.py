import torch
from torchvision import models, transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import numpy as np
# Create preprocessed model
import os
import argparse

parser = argparse.ArgumentParser(description="Example of argparse usage")

parser.add_argument('--arch', type=str, help='Architecture to extract features from')
parser.add_argument('--dataset', type=str, help='Dataset to extract features from')
args = parser.parse_args()

os.environ["TORCH_HOME"] = "~/storage/cache"
#arch = "vit_b_16" # vit_b_16, vit_b_32, vit_l_16, vit_l_32
dataset = args.dataset

constructors = {
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32   
}

model_weights ={
    "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
    "vit_b_32": models.ViT_B_32_Weights.IMAGENET1K_V1,
    "vit_l_16": models.ViT_L_16_Weights.IMAGENET1K_V1,
    "vit_l_32": models.ViT_L_32_Weights.IMAGENET1K_V1   
}


def get_output_path(arch):

    return f"{dataset}/{dataset}_images_feats_{arch}.pth"
    

# data_dir = f"/mnt/nas2/GrimaRepo/araymond//shapes3d_abstraction_{split}_images.npz"
data_dir = f"{args.dataset}/{args.dataset}.pth"

print("Loading dataset...", flush=True)
tensor_images = torch.load(data_dir)['images']
# Data
#tensor_images = torch.from_numpy(images)
ds = TensorDataset(tensor_images)
dl = DataLoader(ds, batch_size=512 if args.dataset == "idsprites" else 256, shuffle=False)
arch = args.arch

# Load a pretrained Vision Transformer model
print(f"Loading pretrained model {args.arch}...", flush=True)
vit_model = constructors[arch](weights=model_weights[arch]).cuda()
# Get the preprocessing transforms for the model
preprocess = model_weights[arch].transforms()
# Run the image through the model
vit_model.heads = torch.nn.Identity() # We want to extract features
vit_model.eval()  # Set the model to evaluation mode

result = []

print("Obtaining representations...", flush=True)
with torch.no_grad():
    for x in tqdm(dl):
        x = x[0].cuda()
        if dataset == "idsprites":
            x = x.repeat(1,3,1,1)
        input_tensor = preprocess(x)  # Add batch dimension
        output = vit_model(input_tensor)
        result.append(output.cpu().detach())

v = torch.vstack(result)    
results_path = get_output_path(arch)
torch.save(v, results_path)

