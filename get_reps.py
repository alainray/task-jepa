import torch
from torchvision import models, transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import numpy as np
# Create preprocessed model
import os

os.environ["TORCH_HOME"] = "~/storage/cache"
arch = "vit_b_16" # vit_b_16, vit_b_32, vit_l_16, vit_l_32
dataset = "idsprites"
splits = ["scale"]
# splits = ['train','shape',"scale",'orientation','x','y']
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

def get_images_path(dataset, split):

    if dataset == "shapes3d":
        return f"/mnt/nas2/GrimaRepo/araymond/shapes3d/shapes3d_abstraction_{split}_images.npz"
    elif dataset == "idsprites":
        return f"/mnt/nas2/GrimaRepo/araymond/idsprites/idsprites_images_{split}.npz"
    else:
        raise ValueError(f"Dataset {dataset} not supported!")

def get_output_path(dataset, split, arch):

    if dataset == "shapes3d":
        return f"/mnt/nas2/GrimaRepo/araymond/shapes3d/shapes3d_abstraction_{split}_images_feats_{arch}.npz"
    elif dataset == "idsprites":
           return f"/mnt/nas2/GrimaRepo/araymond/idsprites/idsprites_images_feats_{split}_{arch}.npz"
    else:
        raise ValueError(f"Dataset {dataset} not supported!")

for split in tqdm(splits):
    # data_dir = f"/mnt/nas2/GrimaRepo/araymond//shapes3d_abstraction_{split}_images.npz"
    data_dir = get_images_path(dataset, split)
    images = np.load(data_dir)['arr_0']
    
    # Data
    tensor_images = torch.from_numpy(images)
    ds = TensorDataset(tensor_images)
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    for arch in tqdm(["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]):
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
        results_path = get_output_path(dataset, split, arch)
        np.savez(results_path, v.numpy())