class LightningTaskJEPA(L.LightningModule):
    def __init__(self, args, encoder, target_encoder):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.criterion = F.smooth_l1_loss
        self.args = args
        self.last_batch = -1 # Records id of last training batch seen
        
    def forward(self, batch):
        x, y = batch
        n_fovs = y.shape[-1]
        # create all pairs of input and target
        latents_zeros = torch.zeros_like(y)
        # Make image pairs (input, target) and assign correct latents
        x_input = x[0]
        x_target = x[1]
        # 4 cases 
        # Same rep:
        # x_0 with latent vs x_1
        # x_0 vs x_1 with -latent
        # Different rep
        # x_0 vs x_1
        # x_0 with latent vs x_1 with latent! 
        targets_1 = self.target_encoder(x_target, latents_zeros)
        output_1 = self.encoder(x_input, y)
        targets_2 = self.encoder(x_target, -y)
        output_2 = self.target_encoder(x_input, latents_zeros)
        return output_1, targets_1, output_2, targets_2

    def on_after_backward(self): # For applying ema
        # Momentum update of target encoder
        
        with torch.no_grad():
            m = next(args.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        
    def training_step(self, batch, batch_idx):

        metrics = dict()
        self.last_batch = batch_idx
        output_1, targets_1, output_2, targets_2 = self(batch)
        same_loss, diff_loss = self.calculate_loss(output_1, targets_1, output_2, targets_2)
        loss = 0
        if "same" == self.args.losses or "all" == self.args.losses:
            loss += same_loss 
            metrics['same_loss'] = loss
        if "latent" == self.args.losses or "all" == self.args.losses:
            loss +=  diff_loss
            metrics['latent_loss'] = loss
        metrics['loss'] = loss
        metrics = {f'train_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def calculate_loss(self, o1, t1, o2, t2, eps=0.01):
        same_loss = self.criterion(o1, t1) + self.criterion(o2, t2)      # Should have same reps
        margin = torch.tensor(1.).to(o1.device) # send margin to correct device
        diff_loss = 1/(torch.min(margin, self.criterion(o1, t2)) + eps)     # Should have different reps
        
        return same_loss, diff_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = dict()
        split = ['val','test'][dataloader_idx]            
        output_1, targets_1, output_2, targets_2 = self(batch)
        loss = 0
        if "same" == self.args.losses or "all" == self.args.losses:
            loss += same_loss 
            metrics['same_loss'] = loss
        if "latent" == self.args.losses or "all" == self.args.losses:
            loss +=  diff_loss
            metrics['latent_loss'] = loss
        metrics['loss'] = loss
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)


    def configure_optimizers(self):
        param_groups = [
                {'params': self.encoder.parameters()}
            ]
        return torch.optim.AdamW(param_groups, lr=self.args.lr)



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
    def __init__(self, root_dir="", pretrained_arch=None, test=False):
        
        super().__init__()
        test = "_test" if test else ""
        data = torch.load(f"{root_dir}/idsprites{test}.pth") # Load full dataset
        #img_path = join(root_dir, f"idsprites_inter_{n_shapes}_images_{split}.npz")
        #latents_path = join(root_dir, f"idsprites_inter_{n_shapes}_latents_{split}.npz")
        reps_path = join(root_dir, f"idsprites_images_feats_{pretrained_arch}.pth")
        self.return_reps = pretrained_arch is not None
        self.images = data['images']
        self.latents = data['latents']
        #self.images = np.load(img_path)['arr_0']
        #self.latents = np.load(latents_path)['arr_0']
        if self.return_reps:
            self.reps = torch.load(reps_path)

        '''if shuffle: # shuffle datasets for validation and testing
            # Generate a random permutation
            # Set the seed
            rng = np.random.default_rng(seed=42)

            # Use the generator's permutation method
            shuffle_indices = rng.permutation(len(self.images))
            
            # Shuffle both arrays using the same indices
            self.images = self.images[shuffle_indices]
            self.latents = self.latents[shuffle_indices]
            if self.return_reps:
                self.reps = self.reps[shuffle_indices]'''
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


class PairedDataset(Dataset):
    def __init__(self, dataset_a, dataset_b, pair_mode='random'):
        """
        Args:
            dataset_a: First dataset (e.g., images from domain A).
            dataset_b: Second dataset (e.g., images from domain B).
            pair_mode: Pairing mode ('random' or 'index').
        """
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.pair_mode = pair_mode
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)
        if hasattr(self.dataset_a, "dataset"):
            self.return_reps = self.dataset_a.dataset.return_reps
        else:
            self.return_reps = self.dataset_a.return_reps

    def __len__(self):
        # Use the smaller of the two datasets to define the length
        return min(self.len_a, self.len_b)

    def __getitem__(self, idx):
        if self.pair_mode == 'random':
            # Random pairing
            if self.return_reps:
                img_a, reps_a, label_a = self.dataset_a[idx]
                random_idx = torch.randint(0, self.len_b, (1,)).item()
                img_b, reps_b, label_b = self.dataset_b[random_idx]
            else:
                img_a, label_a = self.dataset_a[idx]
                random_idx = torch.randint(0, self.len_b, (1,)).item()
                img_b, label_b = self.dataset_b[random_idx]
        elif self.pair_mode == 'index':
            # Index-based pairing
            if self.return_reps:
                img_a, reps_a, label_a = self.dataset_a[idx % self.len_a]
                img_b, reps_b, label_b = self.dataset_b[idx % self.len_b]
            else:
                img_a, label_a = self.dataset_a[idx % self.len_a]
                img_b, label_b = self.dataset_b[idx % self.len_b]
        else:
            raise ValueError("Invalid pair_mode. Use 'random' or 'index'.")

        if self.return_reps:
            return ((img_a, img_b), (reps_a, reps_b), (label_a, label_b))
        else:
            return ((img_a, img_b), (label_a, label_b))


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
    x = [x_input.unsqueeze(1).float() , x_target.unsqueeze(1).float()]
    
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

class LightningStudentRep(L.LightningModule):
    def __init__(self, args, encoder):
        super().__init__()
        self.encoder = encoder
        self.criterion = F.cosine_similarity
        self.args = args
        
    def forward(self, batch): # batch should be x(x0,x1), y (x0-x1) latent diff between 0 and 1
        x, reps_x, latent = batch
        latents_zeros = torch.zeros_like(latent)
        bs = x[0].shape[0]
        x_n = torch.cat((x[0],x[1],x[0],x[1]), dim=0) # for efficiency
        l_n = torch.cat((latent,-latent,latents_zeros,latents_zeros), dim=0) # for efficiency
        reps, encs = self.encoder(x_n, l_n)
    
        return encs[:bs], encs[bs:2*bs], encs[2*bs:3*bs], encs[3*bs:4*bs], reps[0:bs], reps[bs:2*bs], reps_x[0], reps_x[1]

    # We try to replicate the learned representation as a start.
    # We learn to replicate the representation when latents are zero
    # and also learn to replicate it when latents are different to zero.
    def training_step(self, batch, batch_idx):
        split = "train"
        metrics = self.split_step(batch, split)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['train_loss'] # so lightning can train
        
    
    def split_step(self, batch, split):    

        x0_l_enc, x1_l_enc, x0_enc, x1_enc, x0_l_rep, x1_l_rep, x0_gt, x1_gt = self(batch)
        metrics = dict()
        # altered encoded rep must be equal to rep of original image
        latent_loss = 1 - self.criterion(x0_l_enc, x1_enc).mean() 
        latent_loss += 1 - self.criterion(x1_l_enc, x0_enc).mean()
        diff_loss = self.criterion(x0_l_enc, x0_enc).abs().mean()
        diff_loss += self.criterion(x1_l_enc, x1_enc).abs().mean()
        # inner rep must be equal to pretrained ground truth
        same_loss = 1 - self.criterion(x0_l_rep, x0_gt).mean()
        same_loss += 1 - self.criterion(x1_l_rep, x1_gt).mean() 

        loss = 0
        
        if "same" == self.args.losses or "all" == self.args.losses:
            pass#loss += same_loss*0 
            #metrics['same_loss'] = same_loss/2
        if "diff" == self.args.losses or "all" == self.args.losses:
            loss += diff_loss
            metrics['diff_loss'] = diff_loss/2
        if "latent" == self.args.losses or "all" == self.args.losses:
            pass
            #loss +=  latent_loss*0
            #metrics['latent_loss'] = latent_loss/2
        metrics['loss'] = loss#/6 # to normalize total loss in range (0..1)

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = ['val','test'][dataloader_idx]

        return self.split_step(batch, split)
        

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        split = ['iid_ood_shape','iid_ood_scale',
        'iid_ood_orientation','iid_ood_x',
        'iid_ood_y'
        ][dataloader_idx]

        return self.split_step(batch, split)

    def configure_optimizers(self):
        param_groups = [
                {'params': self.encoder.parameters()}
            ]
        return torch.optim.AdamW(param_groups, lr=self.args.lr)


def fig_to_img(fig):
    # Attach the canvas to the figure
    canvas = FigureCanvas(fig)

    # Render the figure into a buffer
    canvas.draw()

    # Convert the rendered buffer to a NumPy array
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))  # Shape (height, width, 3)
    return image

class LightningEncoderERM(L.LightningModule):
    def __init__(self, args, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.preds =  defaultdict(lambda: defaultdict(list))
        self.labels =  defaultdict(lambda: defaultdict(list))
        
    def forward(self, batch):

        x, y = batch

        bs, *_ = x[0].shape
        
        x = torch.cat([x[0] , x[1]], dim=0) # concatenate along batch dimension for efficiency
        features = self.encoder(x) 
        x_1 = features[:bs]
        x_2 = features[bs:]

        features = torch.cat([x_1, x_2], dim=-1)

        output = self.predictor(features)
        
        losses = []
        # decompose loss per task
        for i, fov in enumerate(self.args.fovs_tasks):
            losses.append(self.criterion(output[i], y[:,i]))
        
        return torch.stack(losses), [o.detach().cpu() for o in output]

    def training_step(self, batch, batch_idx):
        expected_output = batch[-1]
        expected_output = expected_output.cpu()
        losses, outputs = self(batch)
        loss = torch.mean(losses)
        metrics = {"train_loss": loss}
        for i, l in enumerate(losses):
            metrics[f"train_{self.args.fovs_tasks[i]}_loss"] = l.cpu().detach().item()
        
        for i, output in enumerate(outputs):
            fov = self.args.fovs_tasks[i]
            preds = output.argmax(dim=1)
            y = expected_output[:,i]
            correct = (preds == y).float().sum()
            total = y.numel()
            acc = correct/total
            metrics[f'train_{fov}_acc'] = acc
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        expected_output = batch[-1]
        expected_output = expected_output.cpu()
        losses, outputs = self(batch)
        loss = torch.mean(losses)
        split = ['val','test'][dataloader_idx]
        metrics = {f"{split}_loss": loss}
        for i, l in enumerate(losses):
            metrics[f"{split}_{self.args.fovs_tasks[i]}_loss"] = l.cpu().detach().item()
        for i, output in enumerate(outputs):
            fov = self.args.fovs_tasks[i]
            preds = output.argmax(dim=1)
            y = expected_output[:,i]
            correct = (preds == y).float().sum()
            total = y.numel()
            acc = correct/total
            metrics[f'{split}_{fov}_acc'] = acc
            
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        expected_output = batch[-1]
        expected_output = expected_output.cpu()
        losses, outputs = self(batch)
        loss = torch.mean(losses)
        split = ["id", "iid_ood_shape", "shape", "iid_ood_scale","scale","iid_ood_orientation", "orientation", "iid_ood_x","x", "iid_ood_y","y"][dataloader_idx]
        metrics = {f"{split}_loss": loss}
        for i, l in enumerate(losses):
            metrics[f"{split}_{self.args.fovs_tasks[i]}_loss"] = l.cpu().detach().item()
        for i, output in enumerate(outputs):
            fov = self.args.fovs_tasks[i]
            preds = output.argmax(dim=1)
            self.preds[split][fov].append(output)
            y = expected_output[:,i]
            self.labels[split][fov].append(y)
            correct = (preds == y).float().sum()
            total = y.numel()
            acc = correct/total
            metrics[f'{split}_{fov}_acc'] = acc
            
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def on_test_epoch_end(self):
        for split, data in self.preds.items():
            for fov, preds in data.items():
                self.preds[split][fov] = torch.cat(preds)
                labels = self.labels[split][fov]
                self.labels[split][fov] = torch.cat(labels).int()
        
        cms = dict()

        for split, data in self.preds.items():
            for fov, preds in data.items():
                labels = self.labels[split][fov]
                confusion_matrix = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=3, threshold=0.05)
                confusion_matrix(preds, labels)
                fig, _ = confusion_matrix.plot(labels=["same", "greater than", "lower than"])
                img = fig_to_img(fig)
                cms[f"{split}_cm_{fov}"] = torch.tensor(confusion_matrix.compute().detach().cpu().numpy().astype(int)).float()
                cms[f"{split}_cm_{fov}_img"] = img
                confusion_matrix = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=3, normalize="true", threshold=0.05)
                confusion_matrix(preds, labels)
                fig, _ = confusion_matrix.plot(labels=["same", "greater than", "lower than"])
                img = fig_to_img(fig)
                cms[f"{split}_cm_{fov}_perc"] = torch.tensor(confusion_matrix.compute().detach().cpu().numpy().astype(int)).float()
                cms[f"{split}_cm_{fov}_imgperc"] = img
                #self.log_dict({f"{split}_cm_{fov}": confusion_matrix_computed},on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

        # Clear all preds
        if not self.args.test:
            print(f"Saving to {self.args.experiment_id}_cm.pth")
            torch.save(cms, f"{self.args.experiment_id}_cm.pth")
        self.preds =  defaultdict(lambda: defaultdict(list)) # Reset validation preds
        self.labels =  defaultdict(lambda: defaultdict(list)) # Reset validation ground truth labels


    def configure_optimizers(self):
        param_groups = [
                {'params': self.encoder.parameters()},
                {'params': self.predictor.parameters()}
            ]
        return torch.optim.AdamW(param_groups, lr=self.args.lr)
class DownsizeTransformer(nn.Module):
    def __init__(self, n_modules=3, hidden_dim = 768, output_dim=16, num_heads=4, num_layers=3):
        super(DownsizeTransformer, self).__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.semantic_encoder = VisionTransformer(
                    image_size=64,
                    patch_size=8,
                    num_layers=4,
                    num_heads=12,
                    hidden_dim=hidden_dim,
                    mlp_dim=128,
                    num_classes=1
                )
        self.semantic_encoder.heads = nn.Identity()
        self.proj = nn.Linear(5, output_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.semantic_encoder.heads = nn.Identity()
        self.semantic_encoder.conv_proj = nn.Conv2d(
                in_channels=1, out_channels=hidden_dim, kernel_size=8, stride=8
            ) # Dataset is black and white
        '''mlps = [nn.Linear(hidden_dim+5, output_dim)] # hidden_dim + 5 latent factors
        for i in range(n_modules-1):
            mlps.append(nn.ReLU())
            mlps.append(nn.Linear(output_dim,output_dim))'''
        encoder_layer = nn.TransformerEncoderLayer(
                        d_model=output_dim,
                        nhead=num_heads, 
                        dim_feedforward=hidden_dim,
                        batch_first=True
                        )
        self.transform_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, l): 
        n = x.shape[0]
        rep = self.semantic_encoder(x)
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        r = rep.view(-1, self.hidden_dim//self.output_dim,self.output_dim)
        l = self.proj(l).view(-1,1,self.output_dim)
        r = torch.cat([r,l],dim=1)
        print(r.shape, batch_class_token.shape)
        r = torch.cat([batch_class_token,r],dim=1)
        #with_latent = torch.cat((rep,l), dim=-1)
        final_rep = self.transform_encoder(r)[:,0]
        return rep, final_rep

        
class SimpleVTransformer(nn.Module):
    def __init__(self, n_modules=3, hidden_dim = 768, output_dim=16, num_heads=4, num_layers=3):
        super(DownsizeTransformer, self).__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.semantic_encoder = VisionTransformer(
                    image_size=64,
                    patch_size=8,
                    num_layers=4,
                    num_heads=12,
                    hidden_dim=hidden_dim,
                    mlp_dim=128,
                    num_classes=1
                )
        self.semantic_encoder.heads = nn.Identity()
        self.proj = nn.Linear(5, output_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.semantic_encoder.heads = nn.Identity()
        self.semantic_encoder.conv_proj = nn.Conv2d(
                in_channels=1, out_channels=hidden_dim, kernel_size=8, stride=8
            ) # Dataset is black and white
        '''mlps = [nn.Linear(hidden_dim+5, output_dim)] # hidden_dim + 5 latent factors
        for i in range(n_modules-1):
            mlps.append(nn.ReLU())
            mlps.append(nn.Linear(output_dim,output_dim))'''
        encoder_layer = nn.TransformerEncoderLayer(
                        d_model=output_dim,
                        nhead=num_heads, 
                        dim_feedforward=hidden_dim,
                        batch_first=True
                        )
        self.transform_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, l): 
        n = x.shape[0]
        rep = self.semantic_encoder(x)
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        r = rep.view(-1, self.hidden_dim//self.output_dim,self.output_dim)
        l = self.proj(l).view(-1,1,self.output_dim)
        r = torch.cat([r,l],dim=1)
        print(r.shape, batch_class_token.shape)
        r = torch.cat([batch_class_token,r],dim=1)
        #with_latent = torch.cat((rep,l), dim=-1)
        final_rep = self.transform_encoder(r)[:,0]
        return rep, final_rep


class PairVisionTransformer(VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            representation_size=representation_size,
            norm_layer=norm_layer,
            conv_stem_configs=conv_stem_configs,
        )
        self.seq_length = (self.seq_length - 1)*2 + 1
        self.encoder = Encoder(
                        self.seq_length,
                        num_layers,
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout,
                        attention_dropout,
                        norm_layer,
                        )
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == 2*self.image_size, f"Wrong image height! Expected {2*self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, 2*h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))

        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.heads(x)

        return x



class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 64x64x3, Output: 64x64x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Input: 64x64x32, Output: 64x64x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Input: 64x64x64, Output: 32x32x128
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Input: 32x32x128, Output: 16x16x256
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # Input: 16x16x256, Output: 8x8x512
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1) # Input: 8x8x512, Output: 4x4x512
        
        # Fully connected layer
        self.fc = nn.Linear(4 * 4 * 512, 384)  # Flatten from 4x4x512 to 384 output
        
    def forward(self, x):
        # Pass through convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 4*4*512)
        
        # Fully connected layer to output the final 384-dimensional vector
        x = self.fc(x)
        
        return x


class LightningTransformRegression(L.LightningModule):
    def __init__(self, args, encoder, modulator,**kwargs):
        super().__init__()
        self.encoder = encoder
        self.use_reps = encoder is None
        self.modulator = modulator
        self.criterion = F.cosine_similarity
        self.args = args
    
    # We try to replicate the learned representation as a start.
    # We learn to replicate the representation when latents are zero
    # and also learn to replicate it when latents are different to zero.
    def training_step(self, batch, batch_idx):
        
        split = "train"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict({k: v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['train_loss'] # so lightning can train


    def forward(self, x, rep_x):

        reps = rep_x if self.use_reps else self.encoder(x.float())     # Image encoding
        bs = x.shape[0]

        l = torch.zeros((bs, len(self.args.FOVS_PER_DATASET)), 
                            dtype=reps.dtype,
                            device=reps.device)
        return self.modulator(reps, l)

    def get_metrics(self, data):
        
        metrics = dict()
        loss = 0
        reg_loss = (data['logits']-data['targets'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.sum(dim=1)                # Average over batch
        loss +=  reg_loss.mean() # TODO: We need to average over tasks.
        n_attrs = len(self.args.FOVS_PER_DATASET)
        dtype = reg_loss.dtype
        device = reg_loss.device
        sum_per_group = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                            data['tasks'],
                                                                            reg_loss,
                                                                            reduce="sum")


        counts = torch.zeros(n_attrs, dtype=data['tasks'].dtype, device=device).scatter_reduce(0, data['tasks'], torch.ones_like(data['tasks']).cuda(), reduce="sum")
        mean_per_group = sum_per_group/counts
        
        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'reg_{task}'] = mean_per_group[i]

        metrics['loss'] = loss
        return metrics 

    def split_step(self, batch):    
        # Batch is simple
        src_img, src_rep, imgs, gt_reps, deltas, src_latents, latents = batch
        bs, n_classes, _ = latents.shape
        #mid_reps = gt_reps if self.use_reps else self.encoder(imgs.float(), gt_reps)     # Image encoding
        src_rep = src_rep.unsqueeze(1).repeat((1,n_classes,1))
        reps = self.modulator(src_rep, deltas)                        # predicted reps given latents
        data = dict()
        data['logits'] = reps.view(bs*n_classes, -1)
        data['targets'] = gt_reps.view(bs*n_classes, -1)
        tasks = deltas.abs().argmax(dim=-1)
        data['tasks'] = tasks.view(-1)
        
        return data

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = "val"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['val_loss']
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        split = "test"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['test_loss']

    def configure_optimizers(self):        
        params = []

        if hasattr(self, 'encoder') and self.encoder is not None:
            params += list(self.encoder.parameters())
        if hasattr(self, 'modulator') and self.modulator is not None:
            params += list(self.modulator.parameters())
        if hasattr(self, 'regressor') and self.regressor is not None:
            params += list(self.regressor.parameters())
        
        param_groups = [{'params': params}]

        return torch.optim.AdamW(param_groups, lr=self.args.lr)



class LightningTransformPlusRegression(L.LightningModule):
    def __init__(self, args, encoder, modulator, regressor,**kwargs):
        super().__init__()
        self.encoder = encoder
        self.use_reps = encoder is None
        self.modulator = modulator
        self.regressor = regressor
        self.criterion = F.cosine_similarity
        self.args = args
    
    # We try to replicate the learned representation as a start.
    # We learn to replicate the representation when latents are zero
    # and also learn to replicate it when latents are different to zero.
    def training_step(self, batch, batch_idx):
        
        split = "train"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict({k: v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['train_loss'] # so lightning can train


    def forward(self, x, rep_x):

        reps = rep_x if self.use_reps else self.encoder(x.float())     # Image encoding
        bs = x.shape[0]

        l = torch.zeros((bs, len(self.args.FOVS_PER_DATASET)), 
                            dtype=reps.dtype,
                            device=reps.device)
        return self.modulator(reps, l)

    def get_regression_loss(self, data):
        
        metrics = dict()
        loss = 0
        
        reg_loss = (data['reg_preds']-data['reg_targets'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.mean(dim=0)                # Average over batch

        loss +=  reg_loss.mean()

        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'lat_reg_{task}'] = reg_loss[i]
            
        metrics['lat_loss'] = loss

        return metrics

    def get_metrics(self, data):
        
        metrics = dict()
        loss = 0
        reg_loss = (data['logits']-data['targets'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.sum(dim=1)                # Average over batch
        loss +=  reg_loss.mean()
        n_attrs = len(self.args.FOVS_PER_DATASET)
        dtype = reg_loss.dtype
        device = reg_loss.device
        sum_per_group = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                            data['tasks'],
                                                                            reg_loss,
                                                                            reduce="sum")


        counts = torch.zeros(n_attrs, dtype=data['tasks'].dtype, device=device).scatter_reduce(0, data['tasks'], torch.ones_like(data['tasks']).cuda(), reduce="sum")
        mean_per_group = sum_per_group/counts
        
        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'reg_{task}'] = mean_per_group[i]

        metrics['loss'] = loss

        m = self.get_regression_loss(data)
        for k, v in m.items(): # copy regression metrics to metrics dict
            metrics[k] = v
        metrics['loss'] += self.args.lambda_latent_loss*m['lat_loss']
        return metrics 

    def split_step(self, batch):    
        # Batch is simple
        src_img, src_rep, imgs, gt_reps, latents = batch
        bs, n_classes, _ = latents.shape
        #mid_reps = gt_reps if self.use_reps else self.encoder(imgs.float(), gt_reps)     # Image encoding
        src_rep = src_rep.unsqueeze(1).repeat((1,n_classes,1))
        reps = self.modulator(src_rep, latents)                        # predicted reps given latents
        data = dict()
        data['logits'] = reps.view(bs*n_classes, -1)
        data['targets'] = gt_reps.view(bs*n_classes, -1)
        tasks = latents.abs().argmax(dim=-1)
        data['tasks'] = tasks.view(-1)
        reg_preds = self.regressor(reps)
        data['reg_preds'] = reg_preds.view(bs*n_classes,-1)
        data['reg_targets'] = latents.view(bs*n_classes,-1)
        
        return data

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = "val"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['val_loss']
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        split = "test"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['test_loss']

    def configure_optimizers(self):        
        params = []

        if hasattr(self, 'encoder') and self.encoder is not None:
            params += list(self.encoder.parameters())
        if hasattr(self, 'modulator') and self.modulator is not None:
            params += list(self.modulator.parameters())
        if hasattr(self, 'regressor') and self.regressor is not None:
            params += list(self.regressor.parameters())
        
        param_groups = [{'params': params}]

        return torch.optim.AdamW(param_groups, lr=self.args.lr)
    
class LatentVisionTransformer(VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        n_latent_attributes: int = 6,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List] = None,
    ):
        super(LatentVisionTransformer, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            representation_size=representation_size,
            norm_layer=norm_layer,
            conv_stem_configs=conv_stem_configs,
        )
        self.n_latent_attributes = n_latent_attributes
        self.latent_proj = nn.Linear(self.n_latent_attributes, hidden_dim)
        # Since we are adding a new token for the latent vector we need to increase sequence length 
        self.seq_length += 1
        self.encoder = Encoder(
            self.seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
    def forward(self, x: torch.Tensor, latent: Optional[torch.Tensor] = None):
        # Reshape and permute the input tensor
        bs = x.shape[0]
        x = self._process_input(x)

        # project latent to hidden_dim
        if latent is None:
            latent = torch.zeros((bs, self.n_latent_attributes)).to(x.device) 
        latent = self.latent_proj(latent.float()).unsqueeze(1)  # (BS, 1, hidden_dim)

        # append latent to sequence
        x = torch.cat((x, latent), dim=1)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


class MLPPredictor(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dims=[1]):
        super(MLPPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.classifiers = nn.Linear(self.hidden_dim, sum(output_dims))
        self.output_dims = output_dims

    def forward(self, x):
        x = self.fc1(x)
        x = self.classifiers(x)
        return torch.split(x, self.output_dims, dim=1)


class MultiHeadClassifier(nn.Module):
    def __init__(self, input_dim=100, output_dims=[1]):
        super(MultiHeadClassifier, self).__init__()
        self.input_dim = input_dim
        self.classifiers = nn.Sequential(nn.Linear(self.input_dim, 128),nn.ReLU(), nn.Linear(128, sum(output_dims)))
        self.output_dims = output_dims

    def forward(self, x):
        x = self.classifiers(x)
        return torch.split(x, self.output_dims, dim=1)

class LightningRepClassification(L.LightningModule):
    def __init__(self, args, encoder, modulator, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.use_reps = encoder is None
        self.modulator = modulator
        self.criterion = F.cosine_similarity
        self.args = args
    
    # We try to replicate the learned representation as a start.
    # We learn to replicate the representation when latents are zero
    # and also learn to replicate it when latents are different to zero.
    def training_step(self, batch, batch_idx):
        
        split = "train"
        data = self.split_step(batch)
        metrics = self.get_metrics(data)
        metrics = {f'{split}_{k}': v for k,v in metrics.items()}
        self.log_dict({k: v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return metrics['train_loss'] # so lightning can train

    def forward(self, x, rep_x):

        reps = rep_x if self.use_reps else self.encoder(x.float())     # Image encoding
        bs = x.shape[0]

        l = torch.zeros((bs, len(self.args.FOVS_PER_DATASET)), 
                            dtype=reps.dtype,
                            device=reps.device)
        return self.modulator(reps, l)
 
    def get_metrics(self, data):
        
        metrics = dict()
        loss = 0
        if "same" ==  self.args.losses or "all" == self.args.losses:
            same_loss = 1 - self.criterion(data['mid_reps'], data['rep_tgt']).mean()
            loss += same_loss
            metrics['same_loss'] = same_loss

        if "class" == self.args.losses or "all" == self.args.losses:
            class_loss = F.cross_entropy(data['logits'], data['class_tgt'], reduction="mean")
            loss +=  class_loss
            metrics['class_loss'] = class_loss

            preds = data['logits'].argmax(dim=-1).view(-1)
            correct = (preds == data['class_tgt']).view(-1).float()

            accuracy = correct.sum()/correct.numel()
            metrics['class_acc'] = accuracy
            dtype=correct.dtype
            device=correct.device
            tasks = data['tasks'].view(-1)
            n_attrs = 5 if self.args.dataset == "idsprites" else 6
            #print(data['tasks'].shape,correct.shape)
            sum_per_group = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                                tasks,
                                                                                correct,
                                                                                reduce="sum")


            counts = torch.zeros(n_attrs, dtype=tasks.dtype, device=device).scatter_reduce(0, tasks, torch.ones_like(tasks).cuda(), reduce="sum")
            mean_per_group = sum_per_group/counts

            for i, task in enumerate(self.args.FOVS_PER_DATASET):
                metrics[f'class_{task}'] = mean_per_group[i]
                
            metrics['loss'] = loss


        #for k, v in metrics.items():
        #    metrics[k] = v.item()
        return metrics 

    def split_step(self, batch):    

        src_img, src_rep, imgs, gt_reps, deltas, src_latents, latents = batch
        zero_latents = torch.zeros_like(deltas)
        deltas_desc = latents.sum(dim=-1)
        bs, n_classes, c, h, w = imgs.shape
        if self.args.encoder.arch == "cnn":
            imgs = imgs.view(bs*n_classes, c, h, w)
        src_rep = src_rep if self.use_reps else self.encoder(src_img.float(), src_rep)     # Image encoding
        mid_reps = gt_reps if self.use_reps else self.encoder(imgs.float(), gt_reps)     # Image encoding
        if self.args.encoder.arch == "cnn":
            mid_reps = mid_reps.view(bs, n_classes, -1)

        src_rep = src_rep.unsqueeze(1).repeat((1,n_classes,1))
        reps = self.modulator(src_rep, deltas)                        # predicted reps given latents
        reps = torch.nn.functional.normalize(reps, p=2.0, dim=1, eps=1e-12)
        tgt_reps = self.modulator(mid_reps, zero_latents)               # reps we are trying to achieve
        tgt_reps = torch.nn.functional.normalize(tgt_reps, p=2.0, dim=1, eps=1e-12)
        logits = torch.matmul(reps, tgt_reps.transpose(1,2)).view(-1, n_classes) # bs x 10 x 10 --> 10bs x 10
        data = dict()
        data['mid_reps'] = mid_reps
        data['rep_tgt'] = gt_reps
        data['logits'] = logits
        targets = torch.tensor(bs*list(range(n_classes))).view(-1, n_classes).to(logits.device)
        tasks = latents.abs().argmax(dim=-1)
        data['class_tgt'] = targets.view(-1)
        data['tasks'] = tasks
        return data



    def configure_optimizers(self):        
        params = []

        if hasattr(self, 'encoder') and self.encoder is not None:
            params += list(self.encoder.parameters())
        if hasattr(self, 'modulator') and self.modulator is not None:
            params += list(self.modulator.parameters())
        if hasattr(self, 'regressor') and self.regressor is not None:
            params += list(self.regressor.parameters())
        
        param_groups = [{'params': params}]

        return torch.optim.AdamW(param_groups, lr=self.args.lr)


class LightningRepClassificationPlus(LightningRepClassification):

    def __init__(self, args, encoder, modulator, regressor, **kwargs):
        super().__init__(args, encoder, modulator)
        self.encoder = encoder
        self.use_reps = encoder is None
        self.regressor = regressor
        self.modulator = modulator
        self.criterion = F.cosine_similarity
        self.args = args

    def get_regression_loss(self, data):
        
        metrics = dict()
        loss = 0
        reg_loss = (data['reg_preds']-data['reg_targets'])**2 # keep loss per dimension for reporting
        reg_loss = reg_loss.mean(dim=0)                # Average over batch

        loss +=  reg_loss.mean()

        for i, task in enumerate(self.args.FOVS_PER_DATASET):
            metrics[f'lat_reg_{task}'] = reg_loss[i]
            
        metrics['lat_loss'] = loss

        return metrics

    def get_metrics(self, data):
        
        metrics = dict()
        loss = 0
        if "same" ==  self.args.losses or "all" == self.args.losses:
            same_loss = 1 - self.criterion(data['mid_reps'], data['rep_tgt']).mean()
            loss += same_loss
            metrics['same_loss'] = same_loss

        if "class" == self.args.losses or "all" == self.args.losses:
            class_loss = F.cross_entropy(data['logits'], data['class_tgt'], reduction="mean")
            loss +=  class_loss
            metrics['class_loss'] = class_loss

            preds = data['logits'].argmax(dim=-1).view(-1)
            correct = (preds == data['class_tgt']).view(-1).float()

            accuracy = correct.sum()/correct.numel()
            metrics['class_acc'] = accuracy
            dtype=correct.dtype
            device=correct.device
            tasks = data['tasks'].view(-1)
            n_attrs = len(self.args.FOVS_PER_DATASET)
            sum_per_group = torch.zeros(n_attrs, dtype=dtype, device=device).scatter_reduce(0,
                                                                                tasks,
                                                                                correct,
                                                                                reduce="sum")

            counts = torch.zeros(n_attrs, dtype=tasks.dtype, device=device).scatter_reduce(0, tasks, torch.ones_like(tasks).cuda(), reduce="sum")
            mean_per_group = sum_per_group/counts

            for i, task in enumerate(self.args.FOVS_PER_DATASET):
                metrics[f'class_{task}'] = mean_per_group[i]
            
            metrics['loss'] = loss

        m = self.get_regression_loss(data)
        for k, v in m.items(): # copy regression metrics to metrics dict
            metrics[k] = v
        metrics['loss'] += self.args.lambda_latent_loss*m['lat_loss']

        return metrics

    def predict(self, batch):
        src_img, src_rep, latents = batch
        
        src_rep = src_rep if self.use_reps else self.encoder(src_img.float(), src_rep)     # Image encoding
        
        zero_latents = torch.zeros_like(latents).float()
        new_reps = self.modulator(src_rep, zero_latents)
        new_reps = torch.nn.functional.normalize(new_reps, p=2.0, dim=1, eps=1e-12)
        data = dict()
        reg_preds = self.regressor(new_reps)
        data['logits'] = reg_preds 
        data['targets'] = latents

        return data

    def split_step(self, batch):    

        src_img, src_rep, imgs, gt_reps, deltas, src_latents, latents = batch
        zero_latents = torch.zeros_like(latents)
        deltas_desc = latents.sum(dim=-1)
        bs, n_classes, c, h, w = imgs.shape

        # Get encoder reps! # input images --> output reps!

        src_rep = self.encode(src_img.float(), src_rep)     # Image encoding
        if self.args.encoder.arch == "cnn":
            imgs = imgs.view(bs*n_classes, c, h, w)
        mid_reps =self.encode(imgs.float(), gt_reps)     # Image encoding
        if self.args.encoder.arch == "cnn":
            mid_reps = mid_reps.view(bs, n_classes, -1)

        src_rep = src_rep.unsqueeze(1).repeat((1,n_classes,1))
        
        # Pass encoder reps through modulator --> input: images and deltas --> output modulated reps and gt reps

        reps = self.modulate(src_rep, deltas)                        # predicted reps given latents
        non_mod_reps = self.modulate(src_rep[:,0].view(bs,-1), zero_latents[:,0].view(bs,-1)) # original reps
        tgt_reps = self.modulate(mid_reps, zero_latents)               # reps we are trying to achieve
     
        # Get classification logits!
        logits = torch.matmul(reps, tgt_reps.transpose(1,2)).view(-1, n_classes) # bs x 10 x 10 --> 10bs x 10
        
        # Register data for 
        data = dict()
        data['mid_reps'] = mid_reps
        data['rep_tgt'] = gt_reps
        data['logits'] = logits
        targets = torch.tensor(bs*list(range(n_classes))).view(-1, n_classes).to(logits.device)
        tasks = deltas.abs().argmax(dim=-1)
        data['class_tgt'] = targets.view(-1)
        data['tasks'] = tasks
        # Regression Task!


        # Add manipulated reps and non-manipulated reps
        # 2 possible regressions!
        #    a. Modulated regression
        #    b. Non Modulated Regression: (baseline)

        # loss: mod regression
        if self.modulator is not None and "mod_reg" in args.losses:
            mod_reps  = reps.view(bs*n_classes,-1)
        reps = torch.cat([mod_reps, non_mod_reps], dim=0)
        tgt_latents = torch.cat([latents.view(bs*n_classes,-1),src_latents.view(bs,-1)], dim=0)
        data['reg_preds'] = self.regressor(reps)
        data['reg_targets'] = tgt_latents
        return data
