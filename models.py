from torchvision.models import VisionTransformer
from torchvision.models.vision_transformer import Encoder
from typing import Optional, Callable, List, NamedTuple
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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
            latent = torch.zeros((bs, 6))    
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

def create_model(args):

    if args.train_method == "erm":
        model = VisionTransformer(
            image_size=64,
            patch_size=8,
            num_layers=4,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=128,
            num_classes=1
        )
        output_dims = [v for fov in args.fovs for k, v in args.n_fovs.items() if fov == k]

        mhc = MultiHeadClassifier(input_dim=384, output_dims=output_dims)
        model.heads = mhc
        return (model,)

    elif args.train_method == "pair_erm": # We try to predict difference in latents
        
        model = PairVisionTransformer(
            image_size=64,
            patch_size=8,
            num_layers=4,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=128,
            num_classes=1
        )
        output_dims = [args.fovs_levels[k] for k in args.fovs_tasks]

        mhc = MultiHeadClassifier(input_dim=384, output_dims=output_dims)
        model.heads = mhc
        return (model,)

    elif "encoder_erm" == args.train_method:
        output_dims = [args.fovs_levels[k] for k in args.fovs_tasks]
        if args.encoder['pretrained_feats']:
            encoder = nn.Identity()
        
        else:
            if args.encoder['arch'] == 'vit':
                encoder = VisionTransformer(
                image_size=64,
                patch_size=8,
                num_layers=4,
                num_heads=12,
                hidden_dim=384,
                mlp_dim=128,
                num_classes=1
                )
                encoder.heads = nn.Identity()
                # load weights
                # turn off weights if required
            elif args.encoder['arch'] == "cnn":
                encoder = SimpleConvModel()
            if args.encoder['frozen']:
                for p in encoder.parameters():
                    p.requires_grad = False
        # get encoder from args
        # input dim is same size as encoder_dim
        predictor = MultiHeadClassifier(input_dim=2*384, output_dims=output_dims)
        #predictor = MLPPredictor(input_dim=2*args.encoder['output_dim'], output_dims=output_dims)
        return encoder, predictor
        
    elif args.train_method in ['task_jepa', 'ijepa']:
        encoder = LatentVisionTransformer(
            image_size=64,
            patch_size=8,
            num_layers=4,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=128,
            num_classes=1
        )
        mhc = nn.Identity()
        encoder.heads = mhc
        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return encoder, target_encoder
