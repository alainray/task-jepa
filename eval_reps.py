import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as f
import torch.nn as nn
from tqdm.notebook import tqdm
from models import LatentVisionTransformer
from train_lightning import get_lightning_model
from utils import set_seed, get_args
import argparse
from train_lightning import LightningStudentRep

parser = argparse.ArgumentParser(description="Example of argparse usage")
# Add arguments
parser.add_argument('--exp_id', type=str, default=None, help='Experiment Id')
parser.add_argument('--num_steps', type=int, default=300000, help='Number of iterations to train')
parser.add_argument('--subsample', type=float, default=1.0, help='Ratio of samples for calculation')

def map_latents_to_values(new_latents_idxs):
    new_latents_idxs = new_latents_idxs.long()
    new_latents_idxs[:,1:] = torch.clamp(new_latents_idxs[:,1:], min=0, max=13)
    latent_indices = f.one_hot(new_latents_idxs[:,1:],num_classes=14)
    new_latents = torch.cat((new_latents_idxs[:,0].unsqueeze(1),(latent_indices*values).sum(dim=-1)),dim=1)
    return new_latents
    
def map_detail(x):
    l = ['shape+','scale+','orientation+','x+','y+',
        'shape-','scale-','orientation-','x-','y-']
    return  l[x]
    
def process_data(df, splits_df):
    new_df = pd.merge(df, splits_df, left_on='original_idx', right_on='idx', how='inner')
    new_df = pd.merge(new_df, splits_df, left_on='gt_idx', right_on='idx', how='inner')
    new_df['detail'] = new_df['detail_pred'].apply(map_detail)
    new_df['setting'] = new_df['split_x'] + "-" + new_df['split_y']
    result = new_df.groupby(['setting', 'delta','attribute','detail'])['correct'].agg(['sum', 'mean','count'])
    result = result.reset_index()
    return result

def run_eval(model, dl):
    model.eval()
    pandas_data = {
                'original_idx': [],
                'gt_idx': [],
                'correct': [], 
                'detail_pred': [], 
                'attribute': [], 
                'delta': [] 
              }
    with torch.no_grad():
        for delta in tqdm([1,2,3,4,5], desc="Running for delta..."):
            for idx, (img, latent, latent_id) in tqdm(enumerate(dl), total=len(dl), desc="Processing data"):
                bs = img.shape[0]
                latent = latent.to("cuda", non_blocking=True, dtype=torch.float32)
                img = img.to("cuda", non_blocking=True, dtype=torch.float32).float()
                # generate delta latents for predictions/false 
                n, *_ = img.shape
                latent_id = latent_id.unsqueeze(1)
                deltas = torch.tensor([delta]*5 + [-delta]*5).repeat(n, 1) # CPU
                deltas = deltas.view(-1)
                attributes = np.tile(np.array(['shape', 'scale', 'orientation','x','y']*2),(n,1))
                attributes = attributes.reshape(-1)
                pred_delta_latents = delta*torch.eye(5)
                pred_delta_latents = torch.cat((pred_delta_latents,-pred_delta_latents),dim=0)
                pred_delta_latents = pred_delta_latents.repeat(n,1,1) # CPU
                
                new_latents_idxs = (latent_id.repeat(1,10,1) + pred_delta_latents).to(torch.int8) # CPU
                out_of_min_range = new_latents_idxs >= 0
                out_of_max_range = new_latents_idxs < 14
                out_of_max_range[:,:,0] = new_latents_idxs[:,:,0] < 54
                out_of_range = out_of_max_range*out_of_min_range
                viable_latents = torch.all(out_of_range,dim=2).view(-1)
                #new_latents_idxs = new_latents_idxs[viable_latents].to(torch.int8) # only keep viable ones
                #deltas = deltas[viable_latents].to(torch.int8)
                #attributes = attributes[viable_latents]
                gt_indices = torch.mm(new_latents_idxs.view(-1,5).float(), idx_coefs)
                gt_indices = gt_indices.long().squeeze()
                gt_indices[gt_indices>2112880-1] = 0
                gt_images = images[gt_indices]
                gt_images = gt_images.to("cuda", non_blocking=True)         # SLOWEST PART!
                new_latents_idxs = new_latents_idxs.view(-1,5).to("cuda",non_blocking=True)
                # Convert latent_ids to latent values! (SLOW)
                #print(new_latents_idxs)
                new_latents = map_latents_to_values(new_latents_idxs)
                new_latents = new_latents.view(n,10,5)
                new_latents_idxs = new_latents_idxs.view(n,10,5)
                pred_delta_latents = new_latents - latent.unsqueeze(1).repeat(1,10,1)
    
                #gt_indices = torch.mm(new_latents_idxs.view(-1,5).float(), idx_coefs)
                #gt_indices = gt_indices.long().squeeze()
                #gt_indices = gt_indices.to("cpu", non_blocking=True)
        
                    # generate latents for +delta for all attributes
                    # generate latents for -delta for all attributes
                    # This should generate between 5 to 10 predictions per image,
                    # as some latents may be outside the dataset range.
                    # assert len(pred_delta_latents) == len(ground_truth_latents)
                # generate latents for ground truth
                
                # retrieve samples for ground truth using latents to get ids
                # calculate reps for predictions and ground truthcd je
                # Create batch for getting reps
                
                n_candidates = 10 # len(pred_delta_latents)
                pred_images = img.repeat(10,1,1,1)
    
                imgs = torch.cat((pred_images,gt_images)).float()
                imgs = imgs.repeat(1,3,1,1)  # Turn to 3 channels for compatibility with models
                zero_latents = torch.zeros_like(pred_delta_latents).float()
                all_latents = torch.cat((pred_delta_latents, zero_latents),dim=0).float().view(-1,5)
                
                # Get reps for candidates and ground truth, batched together for efficiency
                reps = model(imgs, all_latents.cuda())
                reps = reps.view(n,20,-1)
                pred_reps = reps[:, :n_candidates]
                gt_reps = reps[:,n_candidates:]
                # with reps calculate dot product
                pred_reps = nn.functional.normalize(pred_reps, p=2, dim=-1)
                gt_reps = nn.functional.normalize(gt_reps, p=2, dim=-1)
                gt_reps = gt_reps.permute(0,2,1)
                result = torch.matmul(pred_reps, gt_reps).view(n*10,-1)
    
    
                # convert results associated to unviable latents to -infinity
                # so they don't affect classification
                result[~viable_latents] = -torch.inf
                # the ith line is correct if the i-th value is max
                predicted = result.argmax(dim=1)
                ground_truth = torch.arange(0,n_candidates, device=torch.device("cuda"))
                ground_truth = ground_truth.repeat(n,1).view(-1)
                #ground_truth = torch.tensor(list(range(n_candidates))).cuda()
                correct = (predicted == ground_truth).to(torch.int8).cpu()
                detail_pred = predicted.to(torch.int8).cpu() # np.char.add(np.char.add(attributes[predicted.cpu().numpy()], " "), deltas.numpy().astype(str))
                
                # returns
                # img_idx = index for original image
                # ground_truth_indices = index for target ground truth image
                # correct = whether the predicted change is closer to the ground truth representation
                # detail_pred = which was the representation that was predicted 
                # e.g. ("shape/-1", means that it was the change in "shape" with delta -1
                # attributes = which attribute we are evaluating for prediction
                # deltas = what was the delta applied for the prediction
                original_idx = torch.arange(idx*bs, (idx+1)*bs).repeat_interleave(10)
                #original_idx = np.array(list(range()).repeat(10).repeat(2)
                #original_idx = [idx]*n_candidates
                # only add viable examples to result!
                pandas_data['original_idx'].append(original_idx[viable_latents])
                pandas_data['gt_idx'].append(gt_indices[viable_latents])
                pandas_data['correct'].append(correct[viable_latents])
                pandas_data['detail_pred'].append(detail_pred[viable_latents])
                pandas_data['attribute'].append(attributes[viable_latents])
                pandas_data['delta'].append(deltas[viable_latents])
    
                #if idx>5:
                #    break
    pandas_data['original_idx'] = torch.cat(pandas_data['original_idx'],dim=0).numpy()
    pandas_data['gt_idx'] = torch.cat(pandas_data['gt_idx'],dim=0).numpy()
    pandas_data['correct'] = torch.cat(pandas_data['correct'],dim=0).numpy()
    pandas_data['detail_pred'] = torch.cat(pandas_data['detail_pred'],dim=0).numpy()
    pandas_data['attribute'] = np.concatenate(pandas_data['attribute'], axis=0)
    #pandas_data['original_idx'] = torch.cat(pandas_data['original_idx'],dim=0)
    pandas_data['delta'] = torch.cat(pandas_data['delta'],dim=0).numpy()
    
    
    df = pd.DataFrame(pandas_data)
    return df


# Start of main loop

parsed_args = parser.parse_args()


# Load dataset!
print(f"Loading {100*parsed_args.subsample:.0f}% dataset...", flush=True)
data = torch.load("idsprites/idsprites.pth")
images = data['images'].pin_memory()
latents = data['latents']
latent_ids = data['latent_ids']

if parsed_args.subsample < 1.0:
    n = len(images)
    n_indices = int(n*parsed_args.subsample)
    indices = torch.randperm(n)[:n_indices]
    ds = TensorDataset(images[indices], latents[indices], latent_ids[indices])
else:
    ds = TensorDataset(images, latents, latent_ids)

batch_size = 256
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
attrs = ['shape_ids', 'scale', 'orientation', 'x', 'y']
attrs = [data['meta'][k] for k in attrs]
idx_coefs = torch.tensor([14**4,14**3,14**2,14**1,1]).unsqueeze(0).float().T#.to("cuda",non_blocking=True)
values = torch.stack([torch.from_numpy(a) for a in attrs[1:]]).cuda()

# Get experiment data!
exp_id = parsed_args.exp_id#"zarrsri1"

# Define experiment ids for checkpoint
args = get_args(exp_id)
for k,v in vars(parsed_args).items():
    args[k] = v

steps = args.num_steps
# Load Model and weights!
args.encoder['pretrain_method'] = None

ckpt_path = f"results/idsprites/{exp_id}/step={steps}.ckpt"

print(f"Loading model from {ckpt_path}", flush=True)
ckpt = torch.load(ckpt_path)
model = LatentVisionTransformer(
                    image_size=64,
                    patch_size=8,
                    num_layers=4,
                    num_heads=12,
                    hidden_dim=384,
                    mlp_dim=128,
                    num_classes=1,
                    n_latent_attributes = 5
                    )

model.heads = nn.Sequential(nn.ReLU(), nn.Linear(384, 768))
model = LightningStudentRep.load_from_checkpoint(ckpt_path, args=args, encoder=model,strict=True)
model = model.encoder

n_shapes = args.n_shapes
splits_df = pd.read_csv(f"idsprites/split_{n_shapes}.csv")

print(f"Evaluating for {args.exp_id} for {steps} steps.", flush=True)
df = run_eval(model, dl)
processed_df = process_data(df,splits_df)
processed_df.to_csv(f"results/idsprites/{exp_id}_{n_shapes}_{steps}.csv",index=False)