import numpy as np
import torch
from models import get_model_from_exp
from utils import get_args, get_name_from_args
from torch.nn.functional import normalize
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse

def index_to_latent(index):
    latents = []
    for base in latents_bases:
        latents.append(index // base)
        index = index % base
    return np.array(latents)

def latent_to_index(latents):
    return torch.matmul(latents, latents_bases.T.float())

def generate_valid_deltas(base_latent, idx, max_delta=15):
    base_val = int(base_latent[idx])  # ensure it's an integer
    deltas = []

    for d in range(1, max_delta + 1):
        for sign in [-1, 1]:
            new_val = base_val + sign * d
            if 0 <= new_val < latent_dims[idx]:
                delta = torch.zeros(6)
                delta[idx] = sign * d
                deltas.append(delta)

    return torch.stack(deltas) if deltas else torch.zeros(0, 6)

class DeltaDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, latents, reps=None, idx=0):
        self.latents = latents
        self.imgs = imgs if reps is None else reps
        self.idx = idx
        self.use_reps = reps is not None
  
    def __len__(self):
        return len(self.latents)

    def __getitem__(self, index):
        img = self.imgs[index]
        img_shape = img.shape
        img = img.unsqueeze(0)
        latent = self.latents[index]
        deltas = generate_valid_deltas(latent, self.idx)
        mod_latents = latent + deltas
        if self.use_reps:
          img = img.repeat(len(deltas),1)
        else:
          img = img.repeat(len(deltas),1,1,1)
        indices = latent_to_index(mod_latents).long()
        tgt_imgs = self.imgs[indices]
        index = torch.tensor([index]*len(deltas))
        return img, tgt_imgs, deltas, index, indices
        
# Adapt for this use case
def get_dataloader(args, indices = [], bs=1024, shuffle=False,idx=0):
    
    data = torch.load(f"{args.dataset}/{args.dataset}.pth", map_location="cpu")
    reps_path = None
    if args.pretrained_reps:
        reps_path = args.pretrained_reps
    elif args.pretrained_encoder:
        encoder_args = get_args(args.pretrained_encoder)
        reps_path = encoder_args.pretrained_reps
    have_reps = reps_path is not None
    if have_reps:
        print("using pretrained reps...")
        data['reps'] = torch.load(f"{args.dataset}/{args.dataset}_images_feats_{reps_path}.pth", map_location="cpu") if reps_path else None
        data['reps'] = data['reps'] - data['reps'].mean(dim=0) # center
        data['reps'] = torch.nn.functional.normalize(data['reps'], p=2.0, dim=1, eps=1e-12)
    else:
        print("using input images")
    if indices == []:
        indices = torch.tensor([i for i in range(data['images'].shape[0])])
    ds = DeltaDataset(
                data['images'][indices],
                data['latent_ids'][indices],
                data['reps'][indices] if have_reps else data['latents'][indices],
                idx=idx
                )
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return dl


# START OF MAIN LOOP

parser = argparse.ArgumentParser(description="Example of argparse usage")

parser.add_argument('--exp_id', type=str, help='Experiment of model to calculate equivariance for')
input_args = parser.parse_args()


exp_id = input_args.exp_id
# exp_id = "i1h2ptub"
args = get_args(exp_id,update_id=True)
device = "cuda"

# Define basic things for dataset
latents_dims = np.array([10,10,10,8,4,15,1])
latents_bases = torch.tensor(np.concatenate((latents_dims[::-1].cumprod()[::-1][1:],
                                np.array([1,]))))[:-1]
latent_dims = torch.tensor(latents_dims)

# Get model and dataloader for evaluation
model = get_model_from_exp(args).to(device)

test_indices = torch.load(f"{args.dataset}/{args.dataset}_{args.sub_dataset}_test_indices.pth")


eq_by_delta_and_gen = defaultdict(lambda: {
    "abs": [0 for _ in range(4)],  # 4 categories x 6 heads
    "rel": [0 for _ in range(4)],  # 4 categories x 6 heads
    "total": [0 for _ in range(4)]
})


head_names = args.FOVS_PER_DATASET
print("Starting evaluation: ", flush=True)
results = pd.DataFrame()
for idx in range(len(head_names)):
    print(f"Evaluation for latent factor {head_names[idx]}...", flush=True)
    dl = get_dataloader(args, idx=idx)
    ood = torch.zeros(len(dl.dataset))
    ood[test_indices] = 1
    with torch.no_grad():
        for n_batch, (base_imgs, tgt_imgs, deltas, base_idxs, tgt_idxs) in enumerate(tqdm(dl)):
            
            base_imgs = base_imgs.to(device)
            tgt_imgs = tgt_imgs.to(device)
            deltas = deltas.view(-1, deltas.shape[-1])
            deltas = deltas.to(device)
            delta_mags = deltas[:, idx].abs().tolist()
            zero_deltas = torch.zeros_like(deltas).to(device)
            base_idxs = base_idxs.view(-1)
            tgt_idxs = tgt_idxs.view(-1)
            base_reps = model.encode(base_imgs, base_imgs)
            tgt_reps = model.encode(tgt_imgs, tgt_imgs)

            base_reps = base_reps.view(-1, base_reps.shape[-1])
            tgt_reps = tgt_reps.view(-1, tgt_reps.shape[-1])

            # Formula for equivariance from:
            # https://arxiv.org/abs/2211.01244
            # Equations (5) and (6) from section 3.2
            # reps needed for equivariance evaluation
            z_i = model.modulate(tgt_reps, zero_deltas)     # z_i: rep of modified image
            z_0 = model.modulate(base_reps, zero_deltas)    # z_0: rep of base image
            z_i_hat = model.modulate(base_reps, deltas)     # z_i^ : our prediction
        
            # First normalize all reps, as we require cosine similarity
            z_i = normalize(z_i, p=2.0, dim=-1, eps=1e-12)
            z_0 = normalize(z_0, p=2.0, dim=-1, eps=1e-12)
            z_i_hat = normalize(z_i_hat, p=2.0, dim=-1, eps=1e-12)
        
            # Calculate equations from  paper
            eq_term1 = (z_i*z_i_hat).sum(dim=-1)                # sim(z_i, z_i_hat)
            eq_term2 = (z_i*z_0).sum(dim=1)                     # sim(z_i, z_0)
            eq_term1 = torch.clamp(eq_term1, min=-1.0, max=1.0)
            eq_term2 = torch.clamp(eq_term2, min=-1.0, max=1.0)
            
            abs_equivariance = eq_term1 - eq_term2              # Equation (5)
            rel_equivariance =  (1 - eq_term2)/(1-eq_term1)   # Equation (6)

            # Code to calculate statistics later
            for i in range(len(abs_equivariance)):
                delta_mag = int(delta_mags[i])
                source_idx = base_idxs[i]
                target_idx = tgt_idxs[i]
                source_ood = ood[source_idx].item()
                target_ood = ood[target_idx].item()
                category = int(source_ood * 2 + target_ood)

                eq_by_delta_and_gen[delta_mag]["abs"][category] += abs_equivariance[i].cpu().item()
                eq_by_delta_and_gen[delta_mag]["rel"][category] += rel_equivariance[i].cpu().item()
                eq_by_delta_and_gen[delta_mag]["total"][category] += 1

        rows = []
        cat_map = {0: "iid-iid", 1: "iid-ood", 2: "ood-iid", 3: "ood-ood"}
        meta = {
            'dataset': args.dataset,
            'sub_dataset': args.sub_dataset,
            'model': get_name_from_args(args)
        }

        for delta, stats in eq_by_delta_and_gen.items():
            for category in range(4):
                row = {
                    **meta,
                    "delta": delta,
                    "category": cat_map[category],
                    "head": head_names[idx],
                    "abs": stats["abs"][category],
                    "rel": stats["rel"][category],
                    "total": stats["total"][category],
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        results = pd.concat([results, df], ignore_index=True)


results.to_csv(f"results/{args.dataset}/{exp_id}_equivariance.csv")
