import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset, Dataset
from easydict import EasyDict as edict
from models import weights
from utils import set_seed
from functools import partial
from torch.utils.data import Dataset
from os.path import join
from sklearn.model_selection import train_test_split
import random 
import torch.nn.functional as F

class IdSpritesEval(torch.utils.data.Dataset):
    def __init__(self, args, data, indices, max_delta=14, num_samples=2, p_skip=0, test=False, return_indices=False):
        self.return_indices = return_indices
        self.args = args
        self.n_latents = 5 if self.args.dataset == "idsprites" else 6
        if test:
            indices = [0, 38416, 
                2744, 
                196, 
                14, 
                1, 
                76832, 
                5488, 
                392, 
                28, 
                2 ]
        self.p_skip = p_skip
        self.num_samples = num_samples
        self.images = data['images'][indices]
        self.latents = data['latents'][indices] if "latents" in data else torch.empty(len(self.images),1)
        self.latent_ids = data['latent_ids'][indices]
        self.reps = data['reps'][indices] if "reps" in data else torch.empty(len(self.images),1)
        self.pretrain_reps = self.reps.numel() > 0
        if self.pretrain_reps:
            print("Recentering representations!")
            self.reps = self.reps - self.reps.mean(dim=0)
            print("Normalizing reps!")
            self.reps = torch.nn.functional.normalize(self.reps, p=2.0, dim=1, eps=1e-12)

        self.latent_to_idx = {
            tuple(l.tolist()): idx for idx, l in enumerate(self.latent_ids)
        }
        self.meta = data['meta']
        self.indices = indices
        self.max_delta = max_delta 
        # vizable indices are those in the indices
        # self.viable_indices = torch.zeros((len(self.images))).bool()
        # self.viable_indices[indices] = True

    def generate_deltas(self):
        base_delta = [0]*self.n_latents
        #yield tuple(base_delta)
        
        for delta_val in range(1, self.max_delta):
            for idx in range(self.n_latents):
                delta = base_delta.copy()
                delta[idx] = delta_val
                yield tuple(delta)
                
                delta[idx] = -delta_val
                yield tuple(delta)
                
    
    def __getitem__(self, idx):
        latent = self.latent_ids[idx]
        image_ids = []
        deltas = []
        for delta in self.generate_deltas():
            if random.random() < self.p_skip:
                continue
                
            new_latent = latent + torch.tensor(delta)
            new_latent = tuple(new_latent.tolist())
            if not new_latent in self.latent_to_idx:
                continue
                
            image_ids.append(self.latent_to_idx[new_latent])
            deltas.append(delta)
            if len(image_ids) == self.num_samples:
                break
          
        images = self.images[image_ids]
        reps = self.reps[image_ids]
        deltas = torch.tensor(deltas)
        src_image = self.images[idx]
        src_rep = self.reps[idx]
            
        if self.return_indices:
            return idx, src_image, src_rep, images, reps, deltas.float()
        else:

            return src_image, src_rep, images, reps, deltas.float()

    def __len__(self):
        return len(self.latent_ids)

        
def get_indices(args):

    def index_to_latent_id(idx):
        shape = (idx // (14**4)) % 54
        scale =  (idx // (14**3)) % 14
        orientation =  (idx // (14**2)) % 14
        x =  (idx // 14) % 14
        y =  idx % 14
        return (shape,scale,orientation,x,y)
    
    def latent_id_to_split(latent, ood):
        latent = {k: v for k, v in zip(['shape','scale','orientation','x','y'], latent)}
        for k, v in ood.items():
            if latent[k] in v:
                return "ood"
        else:
            return "iid"

    if args.dataset == "idsprites":
        ood = { # defines which attribute ids are out of distribution from training
            'shape': [1,18,35,52],
            'scale': [3,6,9,12],
            'orientation': [3,6,9,12],
            'x': [3,6,9,12],
            'y': [3,6,9,12]
        }
        sample_size = 54 - args.sub_dataset

        if args.sub_dataset == 14:
            ood['shape'] = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13,
            15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33,
            34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        elif args.sub_dataset == 24:
            ood['shape'] = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 16,
              17, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 37, 38,
               41, 42, 44, 46, 48, 50, 51, 52]
        elif args.sub_dataset == 34:
            ood['shape'] = [1, 6, 10, 12, 13, 15, 17, 18, 19, 22,
             24, 25, 26, 29, 30, 33, 34, 35, 39, 41, 46, 47, 49, 52]
        elif args.sub_dataset == 54:
            ood['shape'] = [4, 20, 24, 34]
            
        all_indices = []
        dataset_size = 54*14*14*14*14
        for idx in range(dataset_size):
            latent_id = index_to_latent_id(idx)
            split = latent_id_to_split(latent_id,ood)

            if split == "iid":
                all_indices.append(idx)
    elif args.dataset == "3dshapes":
        all_indices = torch.load(f"3dshapes/shapes3d_{args.sub_dataset}_test_indices.pth") # NOTE: This is NOT a typo! splits seem to be inverted at the origin
        test_indices = torch.load(f"3dshapes/shapes3d_{args.sub_dataset}_train_indices.pth") # not used for training but for post training eval
        train_indices, _ = train_test_split(all_indices, test_size = 0.1, random_state=42)

    return train_indices, all_indices

def load_datasets(args):

    if args.dataset in ["3dshapes", "idsprites"]:
        
        datasets = dict()

        print(f"Loading {args.dataset.capitalize()} dataset...")

        root = args.data_dir
        data = torch.load(f"{args.dataset}/{args.dataset}.pth", map_location="cpu")
        if args.pretrained_reps:
            data['reps'] = torch.load(f"{args.dataset}/{args.dataset}_images_feats_{args.pretrained_reps}.pth", map_location="cpu")
        
        train_indices, val_indices = get_indices(args)
        
        datasets['train'] = IdSpritesEval(
            args,
            data,
            train_indices,
            num_samples=20,
            p_skip=0.0,
            test=args.test
        )

        datasets['val'] = IdSpritesEval(
            args,
            data,
            val_indices,
            num_samples=20,
            p_skip=0.0,
            test=args.test
        )
     
        return datasets
    else:
        raise ValueError(f"Dataset {args.dataset} not supported!")

def get_dataloaders(args, splits=['train','val']):
    ds = load_datasets(args)

    #if args.test:
    #    for ds_name, dataset in ds.items():
    #        ds[ds_name] =  Subset(dataset, list(range(16)))
    
    bs = {
            'erm': {'train': 256, 'val': 1024, 'test': 1024},
            'task_jepa': {'train': 16, 'val': 32, 'test': 32},
            'pair_erm': {'train': 16, 'val': 32, 'test': 32},
            'encoder_erm': {'train': 16, 'val': 32, 'test': 32, 'scale': 32, 'orientation': 32, 'x': 32, 'y': 32},
            'rep_train': {'train': args.train_bs, 'val': args.train_bs, 'test': args.train_bs, 'scale': 32, 'orientation': 32, 'x': 32, 'y': 32} 
         }

    collators = { '3dshapes': {'rep_train': None},
                  'idsprites': {
                        'rep_train': None#partial(create_pairs, mode="diff", dataset="idsprites", train_method=args.train_method)
                    }
    }
    dls = {split : DataLoader(ds[split],
        collate_fn = collators[args.dataset][args.train_method],
        num_workers=args.num_workers,
        persistent_workers= True,
        prefetch_factor = 2,
        pin_memory=True,
        batch_size = bs[args.train_method][split],
        shuffle=split == "train") 
            for split in splits}
    return dls