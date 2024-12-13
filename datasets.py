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

def _load_shapes3d(args, test=False):
    data_dir = args.data_dir
    if args.encoder.arch in weights: # one of the special pretrained architectures
        transform = weights[args.encoder.arch].transforms()
    else:
        transform = transforms.Compose([
        transforms.Normalize((0.5028, 0.5788, 0.6033),
                             (0.3492, 0.4011, 0.4213)),

    ])

    split = 'abstraction'
    if test:
        subsample_train = "_subsample"
        subsample_test = "_subsample"
    else:
        subsample_train = ""
        subsample_test = "_subsample_for_evaluation_in_training" # we use 10% of test data to evaluate during training, then retest on the whole test set.

    if args.pretrained_feats:
        feats_path = f"/mnt/nas2/GrimaRepo/araymond/3dshapes/shapes3d_{split}_train_images{subsample_train}_feats.npz"
        labels_path = f"/mnt/nas2/GrimaRepo/araymond/3dshapes/shapes3d_{split}_train_labels{subsample_train}.npz"
        feats = np.load(feats_path)['arr_0']
        labels = np.load(labels_path)['arr_0']
        # Data
        tensor_images = torch.from_numpy(images)
        tensor_labels = torch.from_numpy(labels).long()
        trainset = TensorDataset(tensor_images, tensor_labels)
        feats_path = f"/mnt/nas2/GrimaRepo/araymond/3dshapes/shapes3d_{split}_test_images{subsample_test}_feats.npz"
        labels_path = f"/mnt/nas2/GrimaRepo/araymond/3dshapes/shapes3d_{split}_test_labels{subsample_test}.npz"
        feats = np.load(feats_path)['arr_0']
        labels = np.load(labels_path)['arr_0']
        # Data
        tensor_images = torch.from_numpy(images)
        tensor_labels = torch.from_numpy(labels).long() 
        testset= TensorDataset(tensor_images, tensor_labels)
        
    else:
        feat = f"_feats_{args.pretrained_reps}" if args.pretrained_reps is not None else ""
        feats_train_path = f'{data_dir}/shapes3d_{split}_train_images{subsample_train}{feat}.npz'
        feats_test_path = f'{data_dir}/shapes3d_{split}_test_images{subsample_test}{feat}.npz'
        trainset = Shapes3DDataset(f'{data_dir}/shapes3d_{split}_train_images{subsample_train}.npz',
                                f'{data_dir}/shapes3d_{split}_train_labels{subsample_train}.npz',
                                transform=transform, shuffle=True,
                                reps_path = feats_train_path)
        testset = Shapes3DDataset(f'{data_dir}/shapes3d_{split}_test_images{subsample_test}.npz',
                                f'{data_dir}/shapes3d_{split}_test_labels{subsample_test}.npz',
                                transform=transform, shuffle=True, 
                                reps_path = feats_test_path)
    return {'train': trainset, 'test': testset}



class Shapes3DDataset:
    """
    floor hue: 10 values linearly spaced in [0, 1]
    wall hue: 10 values linearly spaced in [0, 1]
    object hue: 10 values linearly spaced in [0, 1]
    scale: 8 values linearly spaced in [0, 1]
    shape: 4 values in [0, 1, 2, 3]
    orientation: 15 values linearly spaced in [-30, 30]
    """

    def __init__(self, images_path, latents_path, transform=None, normalize=True, shuffle=False, reps_path=None):
        super().__init__()

        images_files = np.load(images_path)
        self.images = torch.from_numpy(images_files['arr_0'])
        self.images = self.images / 255
        self.return_reps = reps_path is not None
        if self.return_reps:
            self.reps = torch.from_numpy(np.load(reps_path)['arr_0'])
    
        latents_files = np.load(latents_path)
        self.latents = torch.from_numpy(latents_files['arr_0'])
        if normalize:
            self.latents = self.normalize_latents(self.latents)
        if shuffle: # shuffle datasets for validation and testing
            # Generate a random permutation
            # Set the seed
            rng = np.random.default_rng(seed=42)

            # Use the generator's permutation method
            shuffle_indices = rng.permutation(len(self.images))
            
            # Shuffle both arrays using the same indices
            self.images = self.images[shuffle_indices]
            self.latents = self.latents[shuffle_indices]
            if self.return_reps:
                self.reps = self.reps[shuffle_indices]
    
        self.transform = transform

    def normalize_latents(self, array):

        # Calculate the mean and standard deviation for each row (dim=1)
        mean = array.mean(dim=0, keepdim=True)
        std = array.std(dim=0, keepdim=True)
        
        # Normalize the tensor along rows
        normalized_tensor = (array - mean) / std

        return normalized_tensor
        
    def __getitem__(self, idx):
        image, latents = self.images[idx], self.latents[idx]
        if self.transform:
            image = self.transform(image)

        if self.return_reps:
            rep = self.reps[idx]
            return image, rep, latents
        else:
            return image, latents

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return (
            [f'floor_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'wall_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'object_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'scale={idx:.1f}' for idx in np.linspace(0,1,8)]+
            [f'shape={idx}' for idx in range(4)]+
            [f'orientation={int(idx):}' for idx in np.linspace(-30,30,15)]
        )

    @staticmethod
    def task_class_to_str(task_idx, class_idx):
        task_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        classes_per_task = {
            'floor_hue'  : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'wall_hue'   : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'object_hue' : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'scale'      : [f'{idx:.1f}' for idx in np.linspace(0,1,8)],
            'shape'      : list(range(4)),
            'orientation': [f'{idx:.1f}' for idx in np.linspace(-30,30,15)],
        }
        task_name = task_names[task_idx]
        class_value = classes_per_task[task_name][class_idx]
        return f'{task_name}={class_value}'

    @property
    def n_classes_by_latent(self):
        max_value_cls_per_task = self.latents.max(dim=0).values
        n_cls_per_task = max_value_cls_per_task + 1
        return tuple(n_cls_per_task.tolist())

class IdSprites(Dataset):
    def __init__(self, root_dir="", split="train", shuffle=False, pretrained_arch=None):
        super().__init__()
        img_path = join(root_dir, f"idsprites_images_{split}.npz")
        latents_path = join(root_dir, f"idsprites_latents_{split}.npz")
        reps_path = join(root_dir, f"idsprites_images_feats_{split}_{pretrained_arch}.npz")
        self.return_reps = pretrained_arch is not None
        self.images = np.load(img_path)['arr_0']
        self.latents = np.load(latents_path)['arr_0']
        if self.return_reps:
            self.reps = np.load(reps_path)['arr_0']

        if shuffle: # shuffle datasets for validation and testing
            # Generate a random permutation
            # Set the seed
            rng = np.random.default_rng(seed=42)

            # Use the generator's permutation method
            shuffle_indices = rng.permutation(len(self.images))
            
            # Shuffle both arrays using the same indices
            self.images = self.images[shuffle_indices]
            self.latents = self.latents[shuffle_indices]
            if self.return_reps:
                self.reps = self.reps[shuffle_indices]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        latents = self.latents[idx]
        if self.return_reps:
            rep = self.reps[idx]
            return image, rep, latents
        else:
            return image, latents
 

def create_pairs(batch, mode="diff", dataset="3dshapes", train_method="encoder_erm"):   
    len_batch = len(batch[0])
    if len_batch == 2:
        x, y = torch.utils.data.default_collate(batch)
    else:
        x, reps, y = torch.utils.data.default_collate(batch)
    #print(batch[0].shape, batch[1].shape)
    #x, y = batch
    # define pairs and labels for pairwise training
    n_fovs = y.shape[-1]
    # create all pairs of input and target
    B, C, H, W = x.shape
    # only keep pairs where shapes are equal
    
    # Expand dimensions to create all pair combinations of images
    # x.unsqueeze(1) makes the shape (B, 1, C, H, W)
    # x.unsqueeze(0) makes the shape (1, B, C, H, W)
    image_pairs_1 = x.unsqueeze(1).expand(B, B, C, H, W)  # First element of the pairs (B, B, C, H, W)
    image_pairs_2 = x.unsqueeze(0).expand(B, B, C, H, W)  # Second element of the pairs (B, B, C, H, W)
    
    # Stack along a new dimension (2), so each pair is represented by (B, B, 2, C, H, W)
    all_image_pairs = torch.stack((image_pairs_1, image_pairs_2), dim=2)
    latents_diff = y.unsqueeze(1) - y.unsqueeze(0)
    latents_diff = latents_diff.view(-1, n_fovs)
    # relabel diffs for classification
    # first for shape, floor color, wall color, object color, two values = 0 for same, 1 for different

    # Get the relevant columns
    if mode == "classification":
        if dataset == "shapes3d":
            cols = [0, 1, 2, 4]
            # Modify the tensor in place: set values != 0 to 1 for the selected columns
            latents_diff[:, cols] = torch.where(latents_diff[:, cols] != 0, torch.tensor(1), latents_diff[:, cols])
            # For dimensions [3, 5]: set values > 0 to 1 and values < 0 to 2
            latents_diff[:, [3, 5]] = torch.where(latents_diff[:, [3, 5]] > 0, torch.tensor(1), latents_diff[:, [3, 5]])
            latents_diff[:, [3, 5]] = torch.where(latents_diff[:, [3, 5]] < 0, torch.tensor(2), latents_diff[:, [3, 5]])
        
        elif dataset == "idsprites":
            # For dimensions [3, 5]: set values > 0 to 1 and values < 0 to 2
            latents_diff[:, [0, 1, 2, 3, 4]] = torch.where(latents_diff[:, [0, 1, 2, 3, 4]] > 0, torch.tensor(1), latents_diff[:, [0, 1, 2, 3, 4]])
            latents_diff[:, [0, 1, 2, 3, 4]] = torch.where(latents_diff[:, [0, 1, 2, 3, 4]] < 0, torch.tensor(2), latents_diff[:, [0, 1, 2, 3, 4]])

        y = latents_diff.long()
    else:
        y = latents_diff.float() # just differences in latents

    #latents_diff = latents_diff[:,args.fovs_ids].long()
    all_image_pairs = all_image_pairs.view(-1, 2, C, H, W)

    # Make image pairs (input, target) and assign correct latents
    x_input = all_image_pairs[:,0].squeeze()
    x_target = all_image_pairs[:,1].squeeze()
    x = [x_input , x_target]
    
    if len_batch == 2:
        return x, y
    else:
        B, D = reps.shape
        rep_pairs_1 = reps.unsqueeze(1).expand(B, B, D)  # First element of the pairs (B, B, D)
        rep_pairs_2 = reps.unsqueeze(0).expand(B, B, D)  # Second element of the pairs (B, B, D)

        # Stack along a new dimension (2), so each pair is represented by (B, B, 2, C, H, W)
        all_rep_pairs = torch.stack((rep_pairs_1, rep_pairs_2), dim=2)
        all_rep_pairs = all_rep_pairs.view(-1, 2, D)
        rep_input = all_rep_pairs[:,0].squeeze()
        rep_target = all_rep_pairs[:,1].squeeze()
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


def load_datasets(args):

    datasets = dict()
    if args.dataset == "shapes3d":
        return _load_shapes3d(args, test=args.test) # TODO: abstraer esto para que sea parametrizable.

    elif args.dataset == "idsprites":
        # TODO: logic for creating the datasets
        for split in ['train','shape',"scale","orientation","x", "y"]:
            datasets[split if split != "shape" else "test"] = IdSprites(root_dir = args.data_dir,
             split=split,
             shuffle=True,
             pretrained_arch = args.pretrained_arch if args.pretrained_arch is not None else args.pretrained_reps)
        return datasets
    else:
        raise ValueError(f"Dataset {args.dataset} not supported!")

def get_dataloaders(args, splits=['train','val','test']):
    ds = load_datasets(args)

    #ds = {'train': a[0], 'test': a[1]}
    # Define the train-validation split ratio
    train_size = int(0.8 * len(ds['train']))  # 80% for training
    val_size = len(ds['train']) - train_size  # The rest for validation
    
    # Split the dataset into training and validation
    set_seed(42)
    ds['train'], ds['val'] = random_split(ds['train'], [train_size, val_size])

    
    if args.test:
        for ds_name, dataset in ds.items():
            ds[ds_name] =  Subset(dataset, list(range(16)))
    
    bs = {
            'erm': {'train': 256, 'val': 1024, 'test': 1024},
            'task_jepa': {'train': 16, 'val': 32, 'test': 32},
            'pair_erm': {'train': 16, 'val': 32, 'test': 32},
            'encoder_erm': {'train': 16, 'val': 32, 'test': 32, 'scale': 32, 'orientation': 32, 'x': 32, 'y': 32},
            'rep_train': {'train': 16, 'val': 32, 'test': 32} 
         }

    collators = {'shapes3d': {
                             "task_jepa": partial(create_pairs, mode="diff", dataset="shapes3d", train_method=args.train_method),
                             "encoder_erm": partial(create_pairs, mode="classification", dataset="shapes3d", train_method=args.train_method),
                             'rep_train': partial(create_pairs, mode="diff", dataset="shapes3d", train_method=args.train_method)
                             },
                  'idsprites': {
                             'task_jepa': partial(create_pairs, mode="diff", dataset="idsprites", train_method=args.train_method),
                             "encoder_erm": partial(create_pairs, mode="classification", dataset="idsprites", train_method=args.train_method),
                              'rep_train': partial(create_pairs, mode="diff", dataset="idsprites", train_method=args.train_method)
                             }
    }
    dls = {split : DataLoader(ds[split],
        collate_fn = collators[args.dataset][args.train_method],
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size = bs[args.train_method][split],
        shuffle=split == "train") 
            for split in splits}
    return dls