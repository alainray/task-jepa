import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from easydict import EasyDict as edict
from models import weights


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
        feats_path = f"/mnt/nas2/GrimaRepo/araymond/3dshapes/shapes3d_{split}_train__images{subsample_train}_feats.npz"
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

        trainset = Shapes3DDataset(f'{data_dir}/shapes3d_{split}_train_images{subsample_train}.npz',
                                f'{data_dir}/shapes3d_{split}_train_labels{subsample_train}.npz',
                                transform=transform)
        testset = Shapes3DDataset(f'{data_dir}/shapes3d_{split}_test_images{subsample_test}.npz',
                                f'{data_dir}/shapes3d_{split}_test_labels{subsample_test}.npz',
                                transform=transform, shuffle=True)
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

    def __init__(self, images_path, latents_path, transform=None, normalize=True, shuffle=False):
        super().__init__()

        images_files = np.load(images_path)
        self.images = torch.from_numpy(images_files['arr_0'])
        self.images = self.images / 255

        latents_files = np.load(latents_path)
        self.latents = torch.from_numpy(latents_files['arr_0'])
        if normalize:
            self.latents = self.normalize_latents(self.latents)
        if shuffle: # shuffle datasets for validation and testing
            # Generate a random permutation
            shuffle_indices = np.random.permutation(len(self.images))

            # Shuffle both arrays using the same indices
            self.images = self.images[shuffle_indices]
            self.latents = self.latents[shuffle_indices]
    
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


def create_pairs(batch):
    x, y = torch.utils.data.default_collate(batch)
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
    #equal_shape  = latents_diff[:,4] == 0 # for selecting pairs where the shape is the same
    # relabel diffs for classification
    # first for shape, floor color, wall color, object color, two values = 0 for same, 1 for different

    # Get the relevant columns
    cols = [0, 1, 2, 4]
    # Modify the tensor in place: set values != 0 to 1 for the selected columns
    latents_diff[:, cols] = torch.where(latents_diff[:, cols] != 0, torch.tensor(1), latents_diff[:, cols])
    # For dimensions [3, 5]: set values > 0 to 1 and values < 0 to 2
    latents_diff[:, [3, 5]] = torch.where(latents_diff[:, [3, 5]] > 0, torch.tensor(1), latents_diff[:, [3, 5]])
    latents_diff[:, [3, 5]] = torch.where(latents_diff[:, [3, 5]] < 0, torch.tensor(2), latents_diff[:, [3, 5]])

    #for i in range(6):
    #    print(torch.unique(latents_diff[:,i], return_counts=True))

    #latents_diff = latents_diff[:,args.fovs_ids].long()
    all_image_pairs = all_image_pairs.view(-1, 2, C, H, W)
    #all_image_pairs = all_image_pairs[equal_shape]
    # latents_diff = latents_diff[equal_shape]
    # latents_zeros = torch.zeros_like(latents_diff)
    # Make image pairs (input, target) and assign correct latents
    x_input = all_image_pairs[:,0].squeeze()
    x_target = all_image_pairs[:,1].squeeze()
    x = [x_input , x_target]
    y = latents_diff.long()
    
    return x, y

def get_dataloaders(args):
    #data_dir = "/mnt/nas2/GrimaRepo/fidelrio/gradient_based_inference_on_mnist/data/shapes3d"
    a  =_load_shapes3d(args, test=args.test)
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
    dls = {split : DataLoader(ds[split],
        collate_fn = create_pairs,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size = bs[args.train_method][split],
        shuffle=split == "train") 
            for split in ['train','val','test']}
    return dls