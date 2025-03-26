from torchvision.models import VisionTransformer
from model_info import encoder_constructor,encoders, modulators, weights, model_output_dims
from torchvision.models.vision_transformer import Encoder
from typing import Optional, Callable, List, NamedTuple
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import get_args

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU
# It assumes a pair of images concatenated along the sequence dimension

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


# Possible Architectures

# vit_b_16 224 x 224
# vit_b_32 224 x 224
# vit_l_16 224 x 224
# vit_l_32 224 x 224
#

def create_model(args): 
    global weights
    # baseline architectures

    # Define encoder
    # Define modulator
    # Freeze/unfreeze weights
    model_weights = weights[args.encoder.arch] if args.encoder.pretrained else None 
    encoder = encoders[args.encoder.arch](args, weights=model_weights) if args.encoder.arch != "none" else None
    input_dims = model_output_dims[args.encoder.arch] if args.encoder.arch != "none" else  model_output_dims[args.pretrained_reps]

    print("input_dim",input_dims,
          "hidden_dim",args.modulator.hidden_dim)
    modulator = modulators[args.train_method](input_dim=input_dims,
                                              hidden_dim=args.modulator.hidden_dim,
                                              latent_dim = 5 if args.dataset == "idsprites" else 6
                                              )
    '''
    if args.encoder.arch in encoder_constructor:
        # if frozen use pretrained reps instead of freezing encoder which is more expensive.
        if not args.encoder.frozen:

            encoder = model_constructor[args.encoder.arch](weights=model_weights)
        else:
            encoder = None
    else: # Custom Architectures
        if args.pretrained_feats:
            encoder = nn.Sequential(nn.Linear(model_output_dims[args.encoder['arch']], 384), nn.ReLU())
        else:
            # Simple vision transformer
            if args.encoder.arch == "vit":
                encoder = VisionTransformer(
                    image_size=64,
                    patch_size=8,
                    num_layers=4,
                    num_heads=12,
                    hidden_dim=384,
                    mlp_dim=128,
                    num_classes=1
                )
                output_dims = [v for fov in args.fovs for k, v in args.n_fovs.items() if fov == k]

                mhc = MultiHeadClassifier(input_dim=model_output_dims[args.encoder['arch']], output_dims=output_dims)
                encoder.heads = mhc

            elif args.encoder.arch == "pair_vit":
                encoder = model = PairVisionTransformer(
                    image_size=64,
                    patch_size=8,
                    num_layers=4,
                    num_heads=12,
                    hidden_dim=384,
                    mlp_dim=128,
                    num_classes=1
                )
            elif args.encoder.arch == "lvit": # latent vision transformer
                encoder = LatentVisionTransformer(
                    image_size=64,
                    patch_size=8,
                    num_layers=4,
                    num_heads=12,
                    hidden_dim=384,
                    mlp_dim=128,
                    num_classes=1,
                    n_latent_attributes = 6 if args.dataset == "shapes3d" else 5
                    )
            elif args.encoder.arch == "down": # latent vision transformer
                rep_dim = model_output_dims[args.pretrained_reps]
                encoder = DownsizeTransformer(
                                              n_modules = 3,
                                              hidden_dim = rep_dim,
                                              output_dim = args.encoder.enc_dims
                                              )
            else:
                raise ValueError(f"Unexpected architecture for encoder, received: {args.encoder['arch']}")

    # Alter heads depending on training method
    if args.train_method == "pair_erm": # We try to predict difference in latents
        
        output_dims = [args.fovs_levels[args.dataset][k] for k in args.fovs_tasks]
        mhc = MultiHeadClassifier(input_dim=384, output_dims=output_dims)
        if hasattr(encoder, "heads"):
            encoder.heads = mhc
        predictor = None

    elif args.train_method == 'rep_train':
        rep_dim = model_output_dims[args.pretrained_reps]
        if hasattr(encoder, "heads"):
            encoder.heads = nn.Sequential(nn.ReLU(), nn.Linear(model_output_dims[args.encoder.arch], rep_dim))
        predictor = None

    elif args.encoder['pretrain_method'] == "rep_train":
        a = get_args(args.encoder['id']) # get arguments to get pretrain reps
        pretrain_reps = a.pretrained_reps
        rep_dim = model_output_dims[pretrain_reps]
        if hasattr(encoder, "heads"):
            encoder.heads = nn.Sequential(nn.ReLU(), nn.Linear(model_output_dims[args.encoder.arch], rep_dim))
        output_dims = [args.fovs_levels[args.dataset][k] for k in args.fovs_tasks]
        predictor = MultiHeadClassifier(input_dim=2*rep_dim, output_dims=output_dims)

    elif args.train_method == "encoder_erm":

        output_dims = [args.fovs_levels[args.dataset][k] for k in args.fovs_tasks]
        if hasattr(encoder, "heads"):
            encoder.heads = nn.Identity()
        predictor = MultiHeadClassifier(input_dim=2*model_output_dims[args.encoder.arch], output_dims=output_dims)
        
    elif args.train_method in ['task_jepa', 'ijepa']:
        if hasattr(encoder, "heads"):
            encoder.heads = nn.Identity()
        predictor = copy.deepcopy(encoder)
        for p in predictor.parameters():
            p.requires_grad = False

    # Encoder Options
    if args.encoder['id'] is not None: # We want to load weights from previous experiment "id"
        # Path to the weights
        path = f"results/{args.dataset}/{args.encoder['id']}/epoch={args.encoder['epoch']-1}.ckpt"
        #path = f"models/{args.encoder['id']}/{args.encoder['epoch']}.pth"
        print(f"Loading Weights from {path}")

        # Load weights
        # if model trained with rep_train
    
        weights = torch.load(path)['state_dict']
        encoder_weights = {k[8:]: v for k,v in weights.items() if "encoder" in k}
        if len(encoder_weights) > 0:
            print("Loading Encoder Weights from Checkpoint!")
            encoder.load_state_dict(encoder_weights)
        predictor_weights = {k[10:]: v for k,v in weights.items() if "predictor" in k}
        if len(predictor_weights) > 0:
            print("Loading Predictor Weights from Checkpoint!")
            predictor.load_state_dict(predictor_weights)
        # else you should also load the predictor
    '''
    # Freeze/unfreeze parameters
    if args.encoder.arch != "none":
        for p in encoder.parameters():
            p.requires_grad = not args.encoder['frozen']

    return encoder, modulator
