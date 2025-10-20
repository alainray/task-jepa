from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import torch
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from model_info import BigMultiHeadClassifier, BigDecoder
from utils import get_args, get_name_from_args
from models import get_model_from_exp
import argparse


def get_dataloader(args, indices = [], bs=1024, shuffle=False):
    
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
    ds = TensorDataset(
                data['images'][indices]/255.0,
                data['reps'][indices] if have_reps else data['latents'][indices],
                data['latent_ids'][indices]
                )
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return dl

def get_reps_from_model(exp_id):
    
    args = get_args(exp_id,update_id=True)
    args.encoder['pretrain_method'] = None
    model = get_model_from_exp(args)
    # Data loader is simply a TensorDataset of 
    dl = get_dataloader(args)
    # create reps for full dataset for training autoencoder
    reps = get_reps(args, model, dl)
    return reps

def get_reps(args, model, dl):
    reps = []
    with torch.no_grad():
        for img, rep, latents in tqdm(dl):
            bs = rep.shape[0]
            if args.encoder.arch != "none":
                rep = model.encode(img.float().cuda())
            else:
                rep = rep.float().cuda()

            delta = torch.zeros(bs, len(args.FOVS_PER_DATASET)).cuda()
            new_rep = model.modulator(rep, delta)
            reps.append(new_rep.cpu())
        return torch.cat(reps, dim=0)


@torch.no_grad()
def eval_by_delta(model, classifier, dataloader, ood, device="cuda", idx=0):
    def latent_to_index(latents):
        return torch.matmul(latents, latents_bases.T).sum(dim=-1).long()

    model.eval()
    classifier.eval()

    acc_by_delta_and_gen = defaultdict(lambda: {
        "correct": [[0]*6 for _ in range(4)],  # 4 categories x 6 heads
        "total": [[0]*6 for _ in range(4)]
    })

    for n_batch, (reps, target_latents, deltas) in enumerate(tqdm(dataloader)):
        reps = reps.to(device)
        deltas = deltas.to(device)
        target_latents = target_latents.to(device)

        pred_reps = model.modulator(reps, deltas)
        logits = classifier(pred_reps)  # list of 6 (B, num_classes)
        preds = [torch.argmax(logit, dim=1) for logit in logits]
        targets = [target_latents[:, i] for i in range(6)]
        delta_mags = deltas[:, idx].abs().tolist()

        for i in range(len(reps)):
            delta_mag = int(delta_mags[i])
            source_idx = latent_to_index(target_latents[i] - deltas[i])
            target_idx = latent_to_index(target_latents[i])

            source_ood = ood[source_idx].item()
            target_ood = ood[target_idx].item()
            category = int(source_ood * 2 + target_ood)

            for j in range(6):
                if preds[j][i].item() == targets[j][i].item():
                    acc_by_delta_and_gen[delta_mag]["correct"][category][j] += 1
                acc_by_delta_and_gen[delta_mag]["total"][category][j] += 1

    # Convert to DataFrame
    rows = []
    cat_map = {0: "iid-iid", 1: "iid-ood", 2: "ood-iid", 3: "ood-ood"}

    for delta, stats in acc_by_delta_and_gen.items():
        for category in range(4):
            for head in range(6):
                row = {
                    "delta": delta,
                    "category": cat_map[category],
                    "head": head_names[head],
                    "correct": stats["correct"][category][head],
                    "total": stats["total"][category][head],
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df

def generate_valid_deltas(base_latent, idx, max_delta=15):
    base_val = int(base_latent[idx])  # ensure it's an integer
    deltas = [torch.zeros(6)]

    for d in range(1, max_delta + 1):
        for sign in [-1, 1]:
            new_val = base_val + sign * d
            if 0 <= new_val < latent_dims[idx]:
                delta = torch.zeros(6)
                delta[idx] = sign * d
                deltas.append(delta)

    return torch.stack(deltas) if deltas else torch.zeros(0, 6)

def collate_fn_with_targets(batch, idx=0, max_delta=3):
    images_out = []
    targets_out = []
    deltas_out = []

    for image, base_latent in batch:
        base_latent = base_latent.float()
        deltas = generate_valid_deltas(base_latent, idx, max_delta)

        if deltas.shape[0] == 0:
            continue  # skip if no valid deltas

        for delta in deltas:
            target_latent = base_latent + delta
            images_out.append(image)
            targets_out.append(target_latent)
            deltas_out.append(delta)

    return (
        torch.stack(images_out),
        torch.stack(targets_out),
        torch.stack(deltas_out),
    )

# START OF MAIN LOOP

parser = argparse.ArgumentParser(description="Example of argparse usage")

parser.add_argument('--exp_id', type=str, help='Architecture to extract features from')
input_args = parser.parse_args()

latent_dims = [10, 10, 10, 8, 4, 15] # TODO: Change to adapt to dataset, this is 3dshapes
max_delta = 15                       # how far to go from base_latent[i] 
# Get necessary models:
# model: trained model that we want to evaluate, depends on exp_id
# classifier: pretrained_model that is able to classsify correctly the latent attributes
# decoder: specifiic model that decodes representations from model to create images to be evaluated
# by classifier.
exp_id = input_args.exp_id
args = get_args(exp_id, update_id=True)

latents_sizes = torch.tensor([10,10,10,8,4,15])
latents_bases = torch.cat((latents_sizes.flip(0).cumprod(0).flip(0)[1:], torch.tensor([1])))
base_latent = torch.tensor([9, 9, 9, 7, 3, 14])
# Let's load decoder and classifier for evaluation
device="cuda"
hidden = 256 if args.train_method == "linear" else 128
classifier = BigMultiHeadClassifier(d_hidden=hidden, use_encoder=False,num_blocks=4)

filename = f"{exp_id}_classifier_reps.pth"

weights = torch.load(f"results/classifiers/{args.dataset}/{filename}")
classifier.load_state_dict(weights)
classifier = classifier.to(device)

# Decoder
args = get_args(exp_id, update_id=True)
#exp_id = "e99wn9i9"
#weights=torch.load(f"results/decoders/{args.dataset}/full_decoder.pth")
device = "cuda"
# Model to evaluate
model = get_model_from_exp(args).to(device)
# Get pretrained representations from ENCODER
# TODO: Two options, model has no encoder but pretrained reps, or model uses its own encoder to get the reps.

if args.encoder.arch == "none":
    reps_path = None
    if args.pretrained_reps:
        reps_path = args.pretrained_reps
    elif args.pretrained_encoder:
        encoder_args = get_args(args.pretrained_encoder)
        reps_path = encoder_args.pretrained_reps
    reps= torch.load(f"{args.dataset}/{args.dataset}_images_feats_{reps_path}.pth", map_location="cpu") if reps_path else None
    reps = reps - reps.mean(dim=0) # center
    reps = torch.nn.functional.normalize(reps, p=2.0, dim=1, eps=1e-12)
else:
    reps = get_reps_from_model(exp_id)
latents =  torch.load(f"{args.dataset}/{args.dataset}.pth")['latent_ids']
ds = TensorDataset(reps, latents)

# Let's load indices of what's ID and OOD
# Obtener qué imagenes son ID y cuáles OOD
test_indices = torch.load(f"{args.dataset}/{args.dataset}_{args.sub_dataset}_test_indices.pth")
indices = torch.zeros(len(ds))
indices[test_indices] = 1

# Create DataLoaders
FOVS_PER_DATASET = {'3dshapes': ['floor_hue','wall_hue','object_hue','scale','shape', 'orientation'],
                'idsprites': ["shape","scale","orientation","x","y"],
                "dsprites":  ["shape","scale","orientation","x","y"],
                "mpi3d": ['object_color','object_shape','object_size','camera_height','background_color','h_axis','v_axis']
                }
head_names = FOVS_PER_DATASET[args.dataset]

results_acc = pd.DataFrame()
latents_bases = torch.cat((latents_sizes.flip(0).cumprod(0).flip(0)[1:], torch.tensor([1]))).float().to("cuda")

for idx in range(len(latent_dims)):
    dl = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=partial(collate_fn_with_targets, idx=idx, max_delta=15))

    print(f"Evaluating for latent {head_names[idx]}",flush=True)
    #acc_summary, overall_acc = eval_by_delta(model, classifier, decoder, dl, device="cuda", idx=idx) 
    df_delta = eval_by_delta(model, classifier, dl, indices, device="cuda", idx=idx)
    modulated_factor = head_names[idx]  # idx = modulated dim passed to collate_fn

    # Per-delta accuracy

    df_delta["modulated_factor"] = modulated_factor
    df_delta['dataset'] = args.dataset
    df_delta['sub_dataset'] = args.sub_dataset
    df_delta['method'] = get_name_from_args(args)
    df_delta['seed'] = args.seed

    results_acc = pd.concat([results_acc, df_delta], ignore_index=True)
    print(results_acc, flush=True)

    results_acc.to_csv(f"results/{args.dataset}/{exp_id}_acc_detailed_from_reps.csv")