from torchvision.models import VisionTransformer
from torchvision.models.vision_transformer import Encoder
from torchvision import models, transforms
from typing import Optional, Callable, List, NamedTuple
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# encoder.arch defines model! not training method!
model_constructor = {
    "vit_b_16": models.vit_b_16,    
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32
}

weights = {
    "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,    
    "vit_b_32": models.ViT_B_32_Weights.IMAGENET1K_V1,
    "vit_l_16": models.ViT_L_16_Weights.IMAGENET1K_V1,
    "vit_l_32": models.ViT_L_32_Weights.IMAGENET1K_V1
}

model_output_dims = {
    "lvit": 384,
    "vit": 384,
    "vit_b_16": 768,    
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024
}

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU
# It assumes a pair of images concatenated along the sequence dimension

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
        self.latent_proj = nn.Linear(6, hidden_dim)
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
            latent = torch.zeros((bs, 6)).cuda() 
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


def load_encoder(encoder_args):

    if encoder_args['arch'] == "latent_vt":
        encoder = LatentVisionTransformer(
            image_size=64,
            patch_size=8,
            num_layers=4,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=128,
            num_classes=1
        )

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
    # baseline architectures
    if args.encoder.arch in model_constructor:
        model_weights = weights[args.encoder.arch] if args.encoder.pretrained else None 
        encoder = model_constructor[args.encoder.arch](weights=model_weights)
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
                    num_classes=1
                    )
            else:
                raise ValueError(f"Unexpected architecture for encoder, received: {args.encoder['arch']}")

    # Alter heads depending on training method
    if args.train_method == "pair_erm": # We try to predict difference in latents
        
        output_dims = [args.fovs_levels[k] for k in args.fovs_tasks]
        mhc = MultiHeadClassifier(input_dim=384, output_dims=output_dims)
        encoder.heads = mhc
        predictor = None

    elif args.train_method == "encoder_erm":

        output_dims = [args.fovs_levels[k] for k in args.fovs_tasks]
        encoder.heads = nn.Identity()
        predictor = MultiHeadClassifier(input_dim=2*model_output_dims[args.encoder.arch], output_dims=output_dims)
        
    elif args.train_method in ['task_jepa', 'ijepa']:
        
        encoder.heads = nn.Identity()
        predictor = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

    # Encoder Options
    if args.encoder['id'] is not None:
        # Path to the weights
        path = f"models/{args.encoder['id']}/{args.encoder['epoch']}.pth"
        print(f"Loading Weights from {path}")

        # Load weights
        weights = torch.load(path)
        # Check for structural compatibility
        def validate_state_dict(model, state_dict):
            model_keys = set(model.state_dict().keys())
            weight_keys = set(state_dict.keys())
            if model_keys != weight_keys:
                print(f"Warning: State dict mismatch! Model keys: {model_keys}, Weight keys: {weight_keys}")
                return False
            return True

        # Validate and load weights into encoder
        if validate_state_dict(encoder, weights['model_0']):
            encoder.load_state_dict(weights['model_0'])
            print("Encoder weights loaded successfully.")
        else:
            print("Encoder weights not loaded due to mismatch.")

        # Validate and load weights into predictor
        if validate_state_dict(predictor, weights['model_1']):
            predictor.load_state_dict(weights['model_1'])
            print("Predictor weights loaded successfully.")
        else:
            print("Predictor weights not loaded due to mismatch.")

    # Freeze/unfreeze parameters
    for p in encoder.parameters():
        p.requires_grad = not args.encoder['frozen']

    return encoder, predictor
