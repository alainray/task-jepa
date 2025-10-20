from models import LightningRepClassification, LightningTransformRegression, LightningRegression,create_model
from model_info import encoders, modulators, model_output_dims
from utils import set_seed, get_args
import torch
from datasets import IdSpritesEval
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

def get_dataset(args):
    indices = list(range(480000))
    data = torch.load(f"{args.dataset}/{args.dataset}.pth", map_location="cpu")
    if args.pretrained_reps:
        data['reps'] = torch.load(f"{args.dataset}/{args.dataset}_images_feats_{args.pretrained_reps}.pth", map_location="cpu")
    ds = IdSpritesEval(args, data, indices, max_delta=14, num_samples=20, p_skip=0, test=False, return_indices=True)
    return ds
    
def get_dataloader(args, ds, indices):
    sampler = SubsetRandomSampler(indices=indices)
    dl = DataLoader(ds, batch_size=1024, sampler=sampler)
    return dl

def evaluate(model, dataloader):
    model.eval()
    results = []
    split = ['train','id','ood']

    sss = []
    tasks = []
    _ , _ , src_rep, _, _, latents = next(iter(dataloader))
    n_attrs = latents.shape[-1]
    dims = src_rep.shape[-1]

    device = "cuda"
    y_squared = torch.zeros(n_attrs, dims).to(device)
    ys = torch.zeros(n_attrs, dims).to(device)
    
    with torch.no_grad():
        n_batches = 0
        for n_batch, batch in enumerate(tqdm(dataloader)):
            # Unpack index + batch
            idxs, src_img, src_rep, imgs, gt_reps, latents = batch
            idxs, src_img, src_rep, imgs, gt_reps, latents = idxs.cuda(),src_img.cuda(), src_rep.cuda(), imgs.cuda(), gt_reps.cuda(), latents.cuda()

            n_batches+=1
            data = model.split_step((src_img, src_rep, imgs, gt_reps, latents))
            
            sss.append(torch.sum((data['targets'] - data['logits']) ** 2, dim=1))
            tasks.append(data['tasks'])

            # rolling stats
            for i in range(n_attrs):
                y_squared[i] += (data['targets'][data['tasks'] == i]**2).sum(dim=0)
                ys[i] += data['targets'][data['tasks'] == i].sum(dim=0)

        ss_res = torch.cat(sss, dim=0).cuda()
        tasks = torch.cat(tasks, dim=0).long().cuda()
        
        
        n_attrs = 6
        dtype = ss_res.dtype
        device = ss_res.device
        ss_res = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                            tasks,
                                                                            ss_res,
                                                                            reduce="sum")

        counts = torch.zeros(n_attrs, dtype=tasks.dtype, device=device).scatter_reduce(0,tasks, torch.ones_like(tasks).cuda(), reduce="sum").to(device)
        mus = torch.empty(n_attrs, dims).to(device)
        ss_tot = torch.empty(n_attrs).to(device)
        for i in range(n_attrs):
            mus[i] = ys[i]/counts[i]
            ss_tot[i] = (y_squared[i] - 2*ys[i]*mus[i] + mus[i]**2).sum()  # ==> sum_{i=1 in T} (y_i - mu_T)^2

        r2 = 1- ss_res/ss_tot
    return r2

# Define experiment ids for checkpoint

exps = [#'0hdoi4lw', # composition
       #'6dlybv9s', # composition
       # "dbnxlnfv", # interpolation
       # "tt466w4y", # interpolation
        "te4t8cr4", # extrapolation
        "n0vdpha1" # extrapolation
       ]


for exp_id in tqdm(exps):
    args = get_args(exp_id)
    args.encoder['pretrain_method'] = None
    print(args)

    ds = get_dataset(args)
    df = pd.DataFrame()
    for split in tqdm(['train','id','ood']):
        encoder, modulator = create_model(args)
        model = LightningTransformRegression.load_from_checkpoint(checkpoint_path=f"results/{args.dataset}/{exp_id}/last.ckpt", 
                                            args=args, 
                                            encoder=encoder, 
                                            modulator=modulator)
        if split in ['train','id']:
            indices = torch.load(f"3dshapes/shapes3d_{args.sub_dataset}_train_indices.pth")
            train_indices, val_indices = train_test_split(indices, test_size = 0.1, random_state=42)
            indices = train_indices if split == "train" else val_indices
        elif split == "ood":
            indices = torch.load(f"3dshapes/shapes3d_{args.sub_dataset}_test_indices.pth")
        else:
            print("Split not recognized!")
            indices = torch.tensor([]).long()
            
        dl = get_dataloader(args, ds, indices)
        r2 = evaluate(model, dl)
        
        # Store metadata
        model_name = args.pretrained_reps
        if args.pretrained_reps is None:
            if args.pretrained_encoder is not None:
                enc_args = get_args(args.pretrained_encoder)
                model_name = enc_args.pretrained_reps 
        meta = {
            'split': split,
            'dataset': args.dataset,
            'sub_dataset': args.sub_dataset,
            'model': f"{model_name}"
        }
        
        # Create a long-format DataFrame where each row is a (task, r2) pair
        rows = []
        for i, fov in enumerate(args.FOVS_PER_DATASET):
            rows.append({
                **meta,
                'task': fov,
                'r2': r2[i].item()
            })

        # Append to df
        result_df = pd.DataFrame(rows)
        df = pd.concat([df, result_df], ignore_index=True)

    df.to_csv(f"transform_{exp_id}_{args.dataset}_{args.sub_dataset}_{args.pretrained_reps}.csv")