import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from easydict import EasyDict as edict

def _load_shapes3d(data_dir, test=False):
    transform = transforms.Compose([
        transforms.Normalize((0.5028, 0.5788, 0.6033),
                             (0.3492, 0.4011, 0.4213)),
    ])
    split = 'abstraction'
    if test:
        subsample = "_subsample"
    else:
        subsample = ""

    trainset = Shapes3DDataset(f'{data_dir}/shapes3d_{split}_train_images{subsample}.npz',
                               f'{data_dir}/shapes3d_{split}_train_labels{subsample}.npz',
                               transform=transform)
    testset = Shapes3DDataset(f'{data_dir}/shapes3d_{split}_test_images{subsample}.npz',
                              f'{data_dir}/shapes3d_{split}_test_labels{subsample}.npz',
                              transform=transform)
    return trainset, testset


def _load_mpi3d(data_dir):
    transform = transforms.Compose([
        transforms.Normalize((0.5028, 0.5788, 0.6033),
                             (0.3492, 0.4011, 0.4213)),
    ])
    #data_dir = '.'
    split = 'composition'
    trainset = Shapes3DDataset(f'{data_dir}/mpi3d_{split}_train_images.npz',
                               f'{data_dir}/mpi3d_{split}_train_labels.npz',
                               transform=transform)
    testset = Shapes3DDataset(f'{data_dir}/mpi3d_{split}_test_images.npz',
                              f'{data_dir}/mpi3d_{split}_test_labels.npz',
                              transform=transform)
    return trainset, testset


class Shapes3DDataset:
    """
    floor hue: 10 values linearly spaced in [0, 1]
    wall hue: 10 values linearly spaced in [0, 1]
    object hue: 10 values linearly spaced in [0, 1]
    scale: 8 values linearly spaced in [0, 1]
    shape: 4 values in [0, 1, 2, 3]
    orientation: 15 values linearly spaced in [-30, 30]
    """

    def __init__(self, images_path, latents_path, transform=None, normalize=True):
        super().__init__()

        images_files = np.load(images_path)
        self.images = torch.from_numpy(images_files['arr_0'])
        self.images = self.images / 255

        latents_files = np.load(latents_path)
        self.latents = torch.from_numpy(latents_files['arr_0'])
        if normalize:
            self.latents = self.normalize_latents(self.latents)

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

def get_dataloaders(args):
    #data_dir = "/mnt/nas2/GrimaRepo/fidelrio/gradient_based_inference_on_mnist/data/shapes3d"
    a  =_load_shapes3d(args.data_dir, test=args.test)
    #a  =_load_mpi3d(data_dir)
    ds = {'train': a[0], 'test': a[1]}
    # Define the train-validation split ratio
    train_size = int(0.8 * len(ds['train']))  # 80% for training
    val_size = len(ds['train']) - train_size  # The rest for validation
    
    # Split the dataset into training and validation
    train_dataset, val_dataset = random_split(ds['train'], [train_size, val_size])
    test_dataset = ds['test']
    if args.test:
        train_dataset = Subset(train_dataset, list(range(16)))
        val_dataset = Subset(val_dataset, list(range(16)))
        test_dataset = Subset(test_dataset, list(range(16)))
    ds['train'] = train_dataset
    ds['val'] = val_dataset
    ds['test'] = test_dataset
    bs = {
            'erm': {'train': 256, 'val': 1024, 'test': 1024},
            'task_jepa': {'train': 16, 'val': 32, 'test': 32},
            'pair_erm': {'train': 16, 'val': 32, 'test': 32},
            'encoder_erm': {'train': 16, 'val': 32, 'test': 32} 
         }
    dls = {split : DataLoader(ds[split], batch_size = bs[args.train_method][split], shuffle=True) for split in ['train','val','test']}
    return dls