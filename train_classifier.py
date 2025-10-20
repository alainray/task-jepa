from model_info import BigDecoder, BigMultiHeadClassifier
from models import get_model_from_exp
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from easydict import EasyDict as edict
import torch
import argparse
from utils import get_args
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def train_batch(model, batch, optimizer, device):
    model.train()
    if len(batch) == 3:
        x, _,  y = batch  # x: images, y: tuple of 6 tensors (B,)
    else:
        x, y = batch
    x = x.float().to(device)
    y = [y[:, i].to(device) for i in range(y.shape[1])]
    outputs = model(x)  # list of logits per head

    # Calculate losses and accuracies per head
    losses = [F.cross_entropy(logits, targets) for logits, targets in zip(outputs, y)]
    preds = [torch.argmax(logits, dim=1) for logits in outputs]
    accuracies = [(pred == target).float().mean().item() for pred, target in zip(preds, y)]

    # Total loss is the sum
    loss = sum(losses)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "head_losses": [l.item() for l in losses],
        "head_accuracies": accuracies,
        "overall_accuracy": sum(accuracies) / len(accuracies)
    }

def train_model(model, train_loader, optimizer, device, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()

        # Tracking variables
        running_loss = 0.0
        running_accuracies = [0.0] * len(model.output_dims)
        total_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            metrics = train_batch(model, batch, optimizer, device)

            running_loss += metrics["loss"]
            for i, acc in enumerate(metrics["head_accuracies"]):
                running_accuracies[i] += acc
            total_batches += 1
        # Average training metrics
        avg_loss = running_loss / total_batches
        avg_head_accuracies = [acc / total_batches for acc in running_accuracies]
        avg_overall_accuracy = sum(avg_head_accuracies) / len(avg_head_accuracies)

        print(f"Train Loss: {avg_loss:.4f} | Avg Accuracy per Head: {[f'{a:.3f}' for a in avg_head_accuracies]} | Overall: {avg_overall_accuracy:.3f}")

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

# -------- START OF MAIN LOOP ----------------
# Create dataloader
# images and latents!

parser = argparse.ArgumentParser(description="Example of argparse usage")
# Add arguments
parser.add_argument('--exp_id', type=str, help='Resume Experiment Id')
parser.add_argument('--use_exp_decoder', action="store_true", help="Whether to use the experiment's decoder")
parser.add_argument('--from_imgs', type=int, default=0, help="Whether to train from images or representations")
# Parse the arguments
parsed_args = parser.parse_args()  

device = "cuda"

parsed_args.from_imgs = parsed_args.from_imgs == 1

args = get_args(parsed_args.exp_id, update_id=True)

print("Generating reps from pretrained model...")
reps = get_reps_from_model(args.experiment_id)
latents =  torch.load(f"{args.dataset}/{args.dataset}.pth")['latent_ids']
if parsed_args.from_imgs:
    if parsed_args.use_exp_decoder: # train classifier on images generated by decoder for experiment
        # Create Dataloader for this
        decoder_filename = f"results/decoders/{args.dataset}/{args.experiment_id}_decoder.pth"
        weights=torch.load(decoder_filename)

        print(f"Loading Decoder from {decoder_filename}")
        decoder = BigDecoder(d_hidden=128, hidden_dims=[64, 128, 256, 512, 1024], out_dim=3, cifar_cross_entropy=False, end_in_sigmoid=False).to(device)
        decoder.load_state_dict(weights)
        decoder = decoder.to(device)
    
        #reps=enc_reps
        ds_reps = TensorDataset(reps)
        dl_reps = DataLoader(ds_reps, batch_size=1024,shuffle=False)
        imgs = []
        print("Decoding reps into images...")
        with torch.no_grad():
            for (img,) in tqdm(dl_reps):
                img = img.to(device)
                imgs.append(decoder(img).detach().cpu())
            imgs = torch.cat(imgs, dim=0)/255.0
    
        ds = TensorDataset(imgs, latents)
        dl = DataLoader(ds, batch_size=256, shuffle=True)
    else: # use real images
        dl = get_dataloader(args,bs=256,shuffle=True) 
else:
    print("Training from reps")
    ds = TensorDataset(reps, latents)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

# Create Model
if parsed_args.from_imgs:
    model=BigMultiHeadClassifier()
else:
    hidden_dim = 128 if args.train_method != "linear" else 256
    model = BigMultiHeadClassifier(d_hidden=hidden_dim, use_encoder=False,num_blocks=4)
optimizer = Adam(model.parameters(), lr=0.001)
device = "cuda"

train_model(model, dl, optimizer, device, num_epochs=20 if parsed_args.from_imgs else 100)

if parsed_args.from_imgs:
    if not parsed_args.use_exp_decoder:
        filename = "full_classifier.pth"
    else:
        filename = f"{parsed_args.exp_id}_classifier.pth"
else:
    filename = f"{parsed_args.exp_id}_classifier_reps.pth"

torch.save(model.state_dict(), f"results/classifiers/{args.dataset}/{filename}")
