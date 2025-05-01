from torchvision import models, transforms
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F

class LatentMLP(nn.Module):
    def __init__(self, input_dim=100, latent_dim=5, hidden_dim=128, n_blocks=3):
        super(LatentMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(latent_dim,input_dim)
        modules = [nn.Linear(2*self.input_dim, self.hidden_dim)]
        for i in range(n_blocks-1):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x, l):
        l = self.proj(l)
        x = torch.cat((x,l), dim=-1)
        x = self.model(x)
        return x

class LatentResidualMLP(nn.Module):
    def __init__(self, input_dim=100, latent_dim=5, hidden_dim=128, n_blocks=3):
        super(LatentResidualMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(latent_dim,hidden_dim)
        modules = [nn.Linear(self.input_dim, self.hidden_dim)]
        for i in range(n_blocks-1):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x, l):
        l = self.proj(l)
        x = self.model(x)
        return x + l

class MLP(nn.Module):
    def __init__(self, input_dim=100, latent_dim=5, hidden_dim=128, n_blocks=3):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        modules = [nn.Linear(self.input_dim, self.hidden_dim)]
        for i in range(n_blocks-1):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(self.hidden_dim, self.latent_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = self.model(x)
        return x

class BigEncoder(nn.Module):
    def __init__(self, args, d_hidden=256, hidden_dims=[32, 64, 128, 128, 64], avg_pool_pre_fc=False, **kwargs):
        super().__init__()

        encoder_d_hidden = d_hidden

        self.avg_pool_pre_fc = avg_pool_pre_fc
        #d = config.cae_reduce_factor

        modules = []
        if hidden_dims is None:
            # hidden_dims = [16, 32, 64, 64, 64]
            hidden_dims = [32, 64, 128, 256, 512]

        in_channels = 1 if args.dataset == "idsprites" else 3
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(0.1),
                    # nn.MaxPool2d(3),
                    )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*4, encoder_d_hidden)

    def forward(self, x,l=None):
        x = self.encoder(x)
        if self.avg_pool_pre_fc:
            x = F.adaptive_avg_pool2d(x, (1,1)).squeeze()
        else:
            x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def get_features(self, x):
        # output = self.encoder(x).squeeze(-1).squeeze(-1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, latent_dim=5, token_size=32,  num_layers=3, num_heads=16, dropout=0.1, return_sequence=False):
        super().__init__()
        self.token_size = token_size
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.seq_len = input_dim // token_size + latent_dim
        self.latent_dim = latent_dim
        self.d_model = hidden_dim
        self.return_sequence = return_sequence
        # Layers
        self.projection = nn.Linear(latent_dim, self.latent_dim*self.token_size) # Token embedding for each 
        # Project fixed vector to a sequence
        self.to_sequence = nn.Linear(self.token_size, self.hidden_dim)

        # Positional encodings (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, l=None):
        """
        x: (B, input_dim)
        output: (B, seq_len, d_model) if return_sequence else (B, d_model)
        """
        x = x.view(-1, self.input_dim)
        B = x.shape[0]

        # Project latent vector to hidden_dim space
        if l is not None:
            l = l.view(B, -1)
            l = self.projection(l).view(B, self.latent_dim, -1)
        else:
            l = torch.zeros((B, self.latent_dim, self.token_size), device=x.device)
        # Append tokens to end of list
        x = x.view(B,self.seq_len - self.latent_dim, -1)
        x = torch.cat([x, l], dim=1)  # (B, input_dim + latent_dim, token_size)
        # Project to sequence
        x = self.to_sequence(x)  # (B, seq_len, d_model)
        x = x.view(B, self.seq_len, self.d_model)  # (B, seq_len, d_model) 
        # Add positional embedding
        x = x + self.pos_embedding  # (B, seq_len, d_model)
        # Apply transformer encoder
        x = self.encoder(x)  # (B, seq_len, d_model)
        # Return full sequence or mean pooled
        if self.return_sequence:
            return x[:,:self.seq_len - self.latent_dim]
        else:
            # Average over the original tokens, not the appended tokens
            return x[:,:self.seq_len - self.latent_dim].mean(dim=1)  # (B, d_model)
        
class FiLMMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, latent_dim=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim 

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # FiLM generators: one for each modulated layer
        self.film1 = nn.Linear(latent_dim, 2 * hidden_dim)
        self.film2 = nn.Linear(latent_dim, 2 * hidden_dim)

    def apply_film(self, x, gamma, beta):
        # Apply modulation: scale + shift
        return gamma * x + beta

    def forward(self, x, z):
        # FiLM 1
        x = self.fc1(x)
        gamma1, beta1 = self.film1(z).chunk(2, dim=-1)
        gamma1 = gamma1 + 1  # ensures gamma = 1 when z = 0
        beta1 = beta1        # ensures beta = 0 when z = 0
        x = self.apply_film(x, gamma1, beta1)
        x = F.relu(x)

        # FiLM 2
        x = self.fc2(x)
        gamma2, beta2 = self.film2(z).chunk(2, dim=-1)
        gamma2 = gamma2 + 1
        beta2 = beta2
        x = self.apply_film(x, gamma2, beta2)
        x = F.relu(x)

        # Final layer (no FiLM)
        x = self.fc3(x)
        return x

class LinearOperatorMLP(nn.Module):
    def __init__(self, input_dim=100, latent_dim=5, hidden_dim=128, n_blocks=3):
        super(LinearOperatorMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(latent_dim,hidden_dim*hidden_dim)
        self.projB = nn.Linear(hidden_dim, hidden_dim)
        modules = [nn.Linear(self.input_dim, self.hidden_dim)]
        for i in range(n_blocks-2):
            modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            modules.append(nn.ReLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x, l):
        old_shape = x.shape
        x = x.view(-1, self.input_dim)
        l = self.proj(l)
        l = l.view(-1, self.hidden_dim, self.hidden_dim)
        x = self.model(x).unsqueeze(-1)
        # Linear operator line
        #(bs, hidden_dim, hidden_dim) x (bs, hidden_dim, 1) + (bs, hidden_dim)
        x = (l @ x).squeeze() + self.projB(x.squeeze())
        if len(x.shape) == 3:
            x = x.view(old_shape[0], old_shape[1], self.hidden_dim)
        return x
    

class LatentDirectionMLP(nn.Module):
    def __init__(self, input_dim=100, latent_dim=5, hidden_dim=128, n_blocks=3, dir_dim=32):
        super(LatentDirectionMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dir_dim = dir_dim
        # 
        self.proj = nn.Linear(self.hidden_dim, self.latent_dim*self.dir_dim) 
        self.proj_latent_dirs = nn.Parameter(torch.randn(self.latent_dim, self.dir_dim))  # initialized randomly
        self.weight_delta = nn.Parameter(torch.randn(self.latent_dim))  # Weights deltas before adding them to dot product
        
        self.out_proj = nn.Linear(self.latent_dim*self.dir_dim, self.hidden_dim)

        self.projB = nn.Linear(hidden_dim, hidden_dim)
        modules = [nn.Linear(self.input_dim, self.hidden_dim)]
        for i in range(n_blocks-1):
            modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            modules.append(nn.ReLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x, l):
        x = x.view(-1, self.input_dim)
        l = l.view(-1, self.latent_dim)
        x = self.model(x)        
        x = self.proj(x).view(-1, self.latent_dim, self.dir_dim) # divide to each direction individually
        # projection to each learned latent direction
        x = x*self.proj_latent_dirs                        # Get dot products
        weights_sq_norm = self.proj_latent_dirs.norm(dim=1)**2 # norm of learned latent vectors
        dp = x.sum(dim=-1)
        dp += l*self.weight_delta
        dp = dp/weights_sq_norm
        x = dp.unsqueeze(-1)*self.proj_latent_dirs       # actual projection happens here
        x = x.view(-1, self.latent_dim*self.dir_dim)
        x = self.out_proj(x)                             # result is projected to output space
        return x


encoders = {
    "none": None,
    "cnn": BigEncoder,
    "vit_b_16": models.vit_b_16,    
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32
}

modulators = {
    "rep_train": partial(LatentMLP, n_blocks=3),
    "regression": None,
    "transform": partial(LatentMLP, n_blocks=3),
    "transform_plus": partial(LatentMLP, n_blocks=3),
    "rep_train_plus": partial(LatentMLP, n_blocks=3),
    "rep_train_plus_res": partial(LatentResidualMLP, n_blocks=3),
    "rep_train_plus_trans": partial(SimpleTransformer, token_size=64, num_layers=3, num_heads=16, dropout=0.0, return_sequence=False),
    "rep_train_plus_trans": FiLMMLP,
    "rep_train_same": partial(LatentMLP, n_blocks=3),
    "rep_train_same_res": partial(LatentResidualMLP, n_blocks=3),
    "rep_train_same_trans": partial(SimpleTransformer, token_size=64, num_layers=3, num_heads=16, dropout=0.0, return_sequence=False),
    "rep_train_same_film": FiLMMLP,
    "rep_train_same_linop": partial(LinearOperatorMLP, n_blocks=3),
    "rep_train_same_latdir": partial(LatentDirectionMLP, n_blocks=3, dir_dim=32),
    "mod_regression": partial(LatentMLP, n_blocks=3),
    "non_mod_regression": partial(LatentMLP, n_blocks=3),
    "mod_regression_trans": partial(SimpleTransformer, token_size=64, num_layers=3, num_heads=16, dropout=0.0, return_sequence=False),
    "mod_regression_film": FiLMMLP,
    "mod_regression_linop": partial(LinearOperatorMLP, n_blocks=3),
    "mod_regression_latdir": partial(LatentDirectionMLP, n_blocks=3, dir_dim=32)
   
}

regressors = {
    "regression": partial(MLP, n_blocks=3),
    "transform_plus": partial(MLP, n_blocks=3),
    "mod_regression": partial(MLP, n_blocks=3),
    "mod_regression_trans": partial(MLP, n_blocks=3),
    "mod_regression_film": partial(MLP, n_blocks=3),
    "mod_regression_linop": partial(MLP, n_blocks=3),
    "mod_regression_latdir": partial(MLP, n_blocks=3),
    "non_mod_regression": partial(MLP, n_blocks=3),
    "rep_train_plus": partial(MLP, n_blocks=3),
    "rep_train_plus_res": partial(MLP, n_blocks=3),
    "rep_train_plus_trans": partial(MLP, n_blocks=3),
    "rep_train_plus_film": partial(MLP, n_blocks=3),
    "rep_train_same": partial(MLP, n_blocks=3),
    "rep_train_same_res": partial(MLP, n_blocks=3),
    "rep_train_same_trans": partial(MLP, n_blocks=3),
    "rep_train_same_linop": partial(MLP, n_blocks=3),
    "rep_train_same_latdir": partial(MLP, n_blocks=3),
    "rep_train_same_film": partial(MLP, n_blocks=3)

}

encoder_constructor = {
    "cnn": BigEncoder,
    "vit_b_16": models.vit_b_16,    
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32
}

weights = {
    "cnn": None,
    "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,    
    "vit_b_32": models.ViT_B_32_Weights.IMAGENET1K_V1,
    "vit_l_16": models.ViT_L_16_Weights.IMAGENET1K_V1,
    "vit_l_32": models.ViT_L_32_Weights.IMAGENET1K_V1
}

model_output_dims = {
    "cnn": 256,
    "lvit": 384,
    "vit": 384,
    "vit_b_16": 768,    
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024
}

