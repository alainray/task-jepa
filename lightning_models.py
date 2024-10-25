from torchvision.models import VisionTransformer
from torchvision.models.vision_transformer import Encoder
from typing import Optional, Callable, List
from functools import partial
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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
        # Since we are adding a new token for the latent vector we need increase sequence length 
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

class MultiHeadClassifier(nn.Module):
    def __init__(self, input_dim=100, output_dims=[1]):
        super(MultiHeadClassifier, self).__init__()
        self.input_dim = input_dim
        self.classifiers = nn.Linear(self.input_dim, sum(output_dims))
        self.output_dims = output_dims

    def forward(self, x):
        x = self.classifiers(x)
        return torch.split(x, self.output_dims, dim=1)


class ViTLatentClassification(pl.LightningModule):
    def __init__(self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        output_dims: list[int] = [1000],
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List] = None,):

        self.latent_model = LatentVisionTransformer(image_size = image_size,
                                                   patch_size = patch_size,)
        self.classifier = MultiHeadClassifier(hidden_dim = hidden_dim)

    def forward(self, x):
        x = self.latent_model(x)
        return x
        
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
    elif "jepa" in args.train_method:
        model = LatentVisionTransformer(
            image_size=64,
            patch_size=8,
            num_layers=4,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=128,
            num_classes=1
        )
        mhc = nn.Identity()
        model.heads = mhc
        target_encoder = copy.deepcopy(model)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return model, target_encoder
