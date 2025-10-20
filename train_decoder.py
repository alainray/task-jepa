from models import LightningVersatile,create_model
from utils import set_seed, get_args
import torch
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam


class BigDecoder(nn.Module):
    def __init__(self, d_hidden=None, hidden_dims=None, out_dim=3, cifar_cross_entropy=False, end_in_sigmoid=False):
        super().__init__()
        self.cifar_cross_entropy = cifar_cross_entropy

        decoder_d_hidden = d_hidden

        # Build Decoder
        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        self.last_hidden_dim = hidden_dims[-1]

        self.decoder_input = nn.Linear(decoder_d_hidden, self.last_hidden_dim * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=out_dim,
                                      kernel_size= 3, padding= 1),
                            # nn.Tanh())
                            )
        if end_in_sigmoid:
            self.final_layer.append(nn.Sigmoid())

    def forward(self, x):
        result = self.decoder_input(x)
        result = result.view(-1, self.last_hidden_dim, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        if self.cifar_cross_entropy:
            n, c_d, h, w = result.shape
            result = result.reshape(n, -1, 3, h, w)
        return result



def get_reps(args, model, dl):
    reps = []
    with torch.no_grad():
        for img, rep, latents in tqdm(dl):
            bs = rep.shape[0]
            rep = rep.float().cuda()

            delta = torch.zeros(bs, len(args.FOVS_PER_DATASET)).cuda()
            new_rep = model.modulator(rep, delta)
            reps.append(new_rep.cpu())
        return torch.cat(reps, dim=0)

def get_dataloader(args, indices = []):
    
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
                data['images'][indices],
                data['reps'][indices] if have_reps else data['latents'][indices],
                data['latent_ids'][indices]
                )
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False)
    return dl

def train_step(model,
               optimizer,
               batch,
               device,
               process_input,
               oracle_input,
               variational,
               kld_weight,
               cross_entropy=False):
    model.train()

    data, labels = batch

    data = data.to(device)
    labels = labels.to(device)

    if process_input:
        x, y = process_input(data)
    else:
        x = data
        y = labels

    optimizer.zero_grad(set_to_none=True)
    if oracle_input == False:
        output = model(x)
    else:
        output = model(x, labels)

    if variational:
        recons, mu, log_var = output
        loss, recons_loss, kld_loss = model.loss_function(y, recons, mu, log_var, kld_weight=kld_weight)
    elif cross_entropy:
        loss = F.cross_entropy(output, (y*255).squeeze(1).long())
    else:
        loss = F.mse_loss(output, y)

    loss.backward()
    optimizer.step()

    if variational:
        return loss, recons_loss, kld_loss

    return loss

def train_autoencoder_generative(model,
                                 dataloader,
                                 optimizer,
                                 device,
                                 epoch,
                                 variational=False,
                                 test_every=0,
                                 test_fn=None,
                                 oracle_input=False,
                                 process_input=None,
                                 kld_weight=0.1,
                                 cross_entropy=False):

    train_loss = 0
    cum_recons_loss = cum_kld_loss = 0
    all_training_losses = []
    for batch_idx, batch in enumerate(dataloader):
        data, labels = batch
        loss = train_step(model,
                          optimizer,
                          batch,
                          device,
                          process_input,
                          oracle_input,
                          variational,
                          kld_weight,
                          cross_entropy=cross_entropy)

        if variational:
            loss, recons_loss, kld_loss = loss
            cum_recons_loss += recons_loss.item()
            cum_kld_loss += kld_loss.item()

        last_loss = loss.item()
        train_loss += last_loss

        all_training_losses.append(last_loss)

        if batch_idx % 187 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

        current_iter = batch_idx + epoch*len(dataloader)
        if test_every > 0 and current_iter % test_every == 0:
            test_fn(model, current_iter)

    if variational:
        train_loss /= len(dataloader) # loss function already averages over batch size, now average across batches.
        cum_recons_loss /= len(dataloader) # loss function already averages over batch size, now average across batches.
        cum_kld_loss /= len(dataloader) # loss function already averages over batch size, now average across batches.
        print('====> Epoch: {} Average loss: {:.4f} (RECON: {:.4f} + KLD: {:.4f})'
              .format(epoch, train_loss, cum_recons_loss, cum_kld_loss) )
        return (train_loss, cum_recons_loss, cum_kld_loss), all_training_losses
    else:
        train_loss /= len(dataloader) # loss function already averages over batch size, now average across batches.
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss) )
        return train_loss, all_training_losses
    

# MAIN STARTS HERE!

exp_ids = [
    "7wv54s11",
    "5d498fra",
    "oqgowb2e",
    "ixo0hja1",
    "l4oni1tw",
    "0enyz21q",
    "n4hzo2v3",
    "yld6obqe",
    "74mck0lb"
]
for exp_id in exp_ids:
    print(f"Starting experiment {exp_id}", flush=True)
    args = get_args(exp_id)
    args.encoder['pretrain_method'] = None
    print(args)
    df = pd.DataFrame()

    encoder, modulator, regressor, decoder = create_model(args)

    model = LightningVersatile.load_from_checkpoint(checkpoint_path=f"results/{args.dataset}/{exp_id}/last.ckpt", 
                                        args=args, 
                                        encoder=encoder, 
                                        modulator=modulator,
                                        regressor=regressor,
                                        decoder=decoder)


    # Data loader is simply a TensorDataset of 
    dl = get_dataloader(args)
    # create reps for full dataset for training autoencoder
    reps = get_reps(args, model, dl)

    # HANDLE DATASET DEPENDING ON MODEL
    reps_path = None
    if args.pretrained_reps:
        reps_path = args.pretrained_reps
    elif args.pretrained_encoder:
        encoder_args = get_args(args.pretrained_encoder)
        reps_path = encoder_args.pretrained_reps
    have_reps = reps_path is not None
    data =  torch.load(f"{args.dataset}/{args.dataset}.pth")
    imgs=data['images']/255.0
    latents = data['latent_ids']
    print("using pretrained reps...")
    # Create dataloader for training Decoder
    decoder_dl = TensorDataset(reps, imgs)
    decoder_dl = DataLoader(decoder_dl, batch_size=256, shuffle=True)

    # Training code for decoder
    # Create Model
    device = "cuda"
    decoder = BigDecoder(d_hidden=128, hidden_dims=[64, 128, 256, 512, 1024], out_dim=3, cifar_cross_entropy=False, end_in_sigmoid=False).to(device)
    opt = Adam(decoder.parameters(), lr=0.001)

    n_epochs = 10

    for epoch in tqdm(range(n_epochs)):
        train_autoencoder_generative(decoder,
                                    decoder_dl,
                                    opt,
                                    device,
                                    epoch,
                                    variational=False,
                                    test_every=0,
                                    test_fn=None,
                                    oracle_input=False,
                                    process_input=None,
                                    kld_weight=0.1,
                                    cross_entropy=False)
        
    torch.save(decoder.state_dict(), f"results/decoders/{args.dataset}/{exp_id}_decoder.pth")