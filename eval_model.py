from easydict import EasyDict as edict
import lightning as L
import torch
from datasets import get_dataloaders
from train_lightning import get_lightning_model
import argparse
from torch.utils.data import Dataset, random_split, DataLoader
from utils import get_args, set_seed

parser = argparse.ArgumentParser(description="Example of argparse usage")
parser.add_argument('--experiment_id', type=str, default=None, help='Experiment id to calculate metrics')
# Parse the arguments
parsed_args = parser.parse_args()
args = edict()

for k,v in vars(parsed_args).items():
    args[k] = v

def create_iidood_dls(args):
    datasets = dict()
    splits =  ['train','shape',"scale","orientation","x", "y"]
    for split in splits:
                datasets[split] = IdSprites(root_dir = args.data_dir,
                 split=split,
                 shuffle=True,
                 pretrained_arch = args.pretrained_arch if args.pretrained_arch is not None else args.pretrained_reps)
    
    train_size = int(0.8 * len(datasets['train']))  # 80% for training
    val_size = len(datasets['train']) - train_size  # The rest for validation
    
    # Split the dataset into training and validation
    set_seed(42)
    datasets['train'], datasets['val'] = random_split(datasets['train'], [train_size, val_size])
    # Load datasets
    
    # Create paired dataset
    paired_datasets = dict()
    splits =  ['shape',"scale","orientation","x", "y"]
    for split in splits:
        paired_datasets[split] = PairedDataset(datasets['val'], datasets[split], pair_mode='index')
    dls = {f"iid_ood_{k}": DataLoader(ds, batch_size=32, shuffle=False, collate_fn=pairwise_collate) for k, ds in paired_datasets.items()}
    return dls

def pairwise_collate(batch, train_method="encoder_erm"):   
    len_batch = len(batch[0])
    if len_batch == 2:
        x, y = torch.utils.data.default_collate(batch)
    else:
        x, reps, y = torch.utils.data.default_collate(batch)
    #print(batch[0].shape, batch[1].shape)
    #x, y = batch
    # define pairs and labels for pairwise training
    n_fovs = y[0].shape[-1]
    # create all pairs of input and target
    # Concatenate
    x = torch.cat((x[0],x[1]), dim=0)
    y = torch.cat((y[0],y[1]), dim=0)
    B, C, H, W = x.shape
    # only keep pairs where shapes are equal
    
    # Expand dimensions to create all pair combinations of images
    # x.unsqueeze(1) makes the shape (B, 1, C, H, W)
    # x.unsqueeze(0) makes the shape (1, B, C, H, W)
    image_pairs_1 = x.unsqueeze(1).expand(B, B, C, H, W)  # First element of the pairs (B, B, C, H, W)
    image_pairs_2 = x.unsqueeze(0).expand(B, B, C, H, W)  # Second element of the pairs (B, B, C, H, W)
    
    # Stack along a new dimension (2), so each pair is represented by (B, B, 2, C, H, W)
    all_image_pairs = torch.stack((image_pairs_1, image_pairs_2), dim=2)
    latents_diff1 = y.unsqueeze(1) - y.unsqueeze(0)
    #latents_diff = y[1] - y[0]
    #latents_diff = y.unsqueeze(1) - y.unsqueeze(0)
    latents_diff1 = latents_diff1.view(-1, n_fovs)
    # relabel diffs for classification
    # first for shape, floor color, wall color, object color, two values = 0 for same, 1 for different


    # For dimensions [3, 5]: set values > 0 to 1 and values < 0 to 2
    latents_diff1[:, [0, 1, 2, 3, 4]] = torch.where(latents_diff1[:, [0, 1, 2, 3, 4]] > 0, torch.tensor(1), latents_diff1[:, [0, 1, 2, 3, 4]])
    latents_diff1[:, [0, 1, 2, 3, 4]] = torch.where(latents_diff1[:, [0, 1, 2, 3, 4]] < 0, torch.tensor(2), latents_diff1[:, [0, 1, 2, 3, 4]])

    y = latents_diff1.long()
   
    all_image_pairs = all_image_pairs.view(-1, 2, C, H, W)
    # Make image pairs (input, target) and assign correct latents
    x_input = all_image_pairs[:,0].squeeze()
    x_target = all_image_pairs[:,1].squeeze()
    # Make image pairs (input, target) and assign correct latents

    # Let's filter x and y
    N, *_ = x_input.shape
    x_input1 =  torch.cat([x_input[:int(N/2)][i:i+B] for i in range(B, N, 2*B)]) # filter pairs that come from the same dataset
    x_input2 =  torch.cat([x_input[int(N/2):][i:i+B] for i in range(0, N, 2*B)]) # filter pairs that come from the same dataset
    x_target1 =  torch.cat([x_target[:int(N/2)][i:i+B] for i in range(B, N, 2*B)]) # filter pairs that come from the same dataset
    x_target2 =  torch.cat([x_target[int(N/2):][i:i+B] for i in range(0, N, 2*B)]) # filter pairs that come from the same dataset
    x_input = torch.cat((x_input1,x_input2),dim=0)
    x_target= torch.cat((x_target1,x_target2),dim=0)
    x = [x_input , x_target]
    y1 = torch.cat([y[:int(N/2)][i:i+B] for i in range(B, N, 2*B)]) # filter pairs that come from the same dataset
    y2 =  torch.cat([y[int(N/2):][i:i+B] for i in range(0, N, 2*B)]) # filter pairs that come from the same dataset
    #print(x_input.shape,y1.shape,y2.shape)
    y = torch.cat((y1,y2),dim=0).long()
     # If we have to handle representations as input
    if len_batch == 2:
        return x, y
    else:

        reps = torch.cat((reps[0],reps[1]), dim=0)
        B, D = reps.shape
        rep_pairs_1 = reps.unsqueeze(1).expand(B, B, D)  # First element of the pairs (B, B, D)
        rep_pairs_2 = reps.unsqueeze(0).expand(B, B, D)  # Second element of the pairs (B, B, D)

        # Stack along a new dimension (2), so each pair is represented by (B, B, 2, C, H, W)
        all_rep_pairs = torch.stack((rep_pairs_1, rep_pairs_2), dim=2)
        all_rep_pairs = all_rep_pairs.view(-1, 2, D)
        rep_input = all_rep_pairs[:,0].squeeze()
        rep_target = all_rep_pairs[:,1].squeeze()
        rep_input1 =  torch.cat([rep_input[:int(N/2)][i:i+B] for i in range(B, N, 2*B)]) # filter pairs that come from the same dataset
        rep_input2 =  torch.cat([rep_input[int(N/2):][i:i+B] for i in range(0, N, 2*B)]) # filter pairs that come from the same dataset
        rep_target1 =  torch.cat([rep_target[:int(N/2)][i:i+B] for i in range(B, N, 2*B)]) # filter pairs that come from the same dataset
        rep_target2 =  torch.cat([rep_target[int(N/2):][i:i+B] for i in range(0, N, 2*B)]) # filter pairs that come from the same dataset
        # Filter elements that come from the same splits
        rep_input = torch.cat((rep_input1,rep_input2),dim=0)
        rep_target= torch.cat((rep_target1,rep_target2),dim=0)
        
        reps = [rep_input , rep_target]

    if train_method in ["encoder_erm", "task_jepa"] :
        if len_batch == 3:
            return reps, y
        else:
            return x, y
    elif train_method == "rep_train":
        return x, reps, y
    else:
        return x, y
    
    return x, y

# Load model
exp_id = args.experiment_id

# Define experiment ids for checkpoint
args = get_args(exp_id)
args.encoder['pretrain_method'] = None
args.test = False
splits =[ "val",
          "iid_ood_shape", "test",
          "iid_ood_scale","scale",
          "iid_ood_orientation", "orientation",
          "iid_ood_x","x",
          "iid_ood_y","y"]

print("Loading DataLoaders")
iidood_dls = create_iidood_dls(args)
dls = get_dataloaders(args, splits=splits)
for k, dl in iidood_dls.items():
    dls[k] = dl
    
ckpt_path = f"results/idsprites/{exp_id}/epoch=49.ckpt"

print("Loading Lightning Model")
model = get_lightning_model(args)

print(f"Loading Tester for experiment {args.experiment_id}")
tester = L.Trainer(max_epochs=1,
                   accelerator="gpu",
                   devices=1)

print("Commencing test!")
tester.test(
    model,
    ckpt_path=ckpt_path,
    dataloaders=[dls[split] for split in splits]
)

print("Finished Testing!")
torch.save(tester.callback_metrics, f"results_{exp_id}.pth")