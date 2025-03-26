import torch
from torchvision import models, transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import numpy as np
# Create preprocessed model
import os
import argparse

parser = argparse.ArgumentParser(description="Example of argparse usage")

parser.add_argument('--n_shapes', type=int, default=14, help='Number of shapes for the generated dataset')

args = parser.parse_args()

os.environ["TORCH_HOME"] = "~/storage/cache"
arch = "vit_b_16" # vit_b_16, vit_b_32, vit_l_16, vit_l_32
dataset = "idsprites"
splits = [ 'train','shape',"scale",
		'orientation','x','y']
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

def get_images_path(dataset, feat_args):
    split = feat_args['split']
    arch = feat_args['arch']
    n_shapes = feat_args['n_shapes'] if 'n_shapes' in feat_args else None
    polation = feat_args['polation'] if 'polation' in feat_args else None
    
    if dataset == "shapes3d":
        return f"/mnt/nas2/GrimaRepo/araymond/shapes3d/shapes3d_abstraction_{split}_images.npz"
    elif dataset == "idsprites":
        return f"idsprites/idsprites_{polation}_{n_shapes}_images_{split}.npz"
    else:
        raise ValueError(f"Dataset {dataset} not supported!")

def get_output_path(dataset, feat_args):

    split = feat_args['split']
    arch = feat_args['arch']
    n_shapes = feat_args['n_shapes'] if 'n_shapes' in feat_args else None
    polation = feat_args['polation'] if 'polation' in feat_args else None
    
    if dataset == "shapes3d":
        return f"/mnt/nas2/GrimaRepo/araymond/shapes3d/shapes3d_abstraction_{split}_images_feats_{arch}.npz"
    elif dataset == "idsprites":
           return f"idsprites/idsprites_{polation}_{n_shapes}_images_feats_{split}_{arch}.npz"
    else:
        raise ValueError(f"Dataset {dataset} not supported!")

feat_args = {'split': None, 'arch': None, 'polation': 'inter', 'shapes': None}

for split in tqdm(splits):
    feat_args['split'] = split
    
    for n_shapes in tqdm([args.n_shapes]): #,24,34,54]):
        feat_args['n_shapes'] = n_shapes
        # data_dir = f"/mnt/nas2/GrimaRepo/araymond//shapes3d_abstraction_{split}_images.npz"
        data_dir = get_images_path(dataset, feat_args)
        images = np.load(data_dir)['arr_0']
        # Data
        tensor_images = torch.from_numpy(images)
        ds = TensorDataset(tensor_images)
        dl = DataLoader(ds, batch_size=512, shuffle=False)
        
        for arch in tqdm(["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]):
            feat_args['arch'] = arch
            # Load a pretrained Vision Transformer model
            vit_model = constructors[arch](weights=model_weights[arch]).cuda()
            # Get the preprocessing transforms for the model
            preprocess = model_weights[arch].transforms()
            # Run the image through the model
            vit_model.heads = torch.nn.Identity() # We want to extract features
            vit_model.eval()  # Set the model to evaluation mode
    
            result = []
            
            with torch.no_grad():
                for x in tqdm(dl):
                    x = x[0].cuda()
                    input_tensor = preprocess(x)  # Add batch dimension
                    output = vit_model(input_tensor)
                    result.append(output.cpu().detach())
            
            v = torch.vstack(result)    
            results_path = get_output_path(dataset, feat_args)
            np.savez(results_path, v.numpy())
