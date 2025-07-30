from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dimension=n_track*2*2 #1 sample contains left cordinate(x1,y1) and right cooridnate (x2,y2) and we have n track input cordinates for each left and right 
        hidden_dimension=256
        output_dimension=n_waypoints*2#each waypoint is 2d cord
        self.mlpplanner=nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        x = torch.cat([track_left, track_right], dim=1)
        x = x.view(x.shape[0], -1)
        output = self.mlpplanner(x)
        return output.view(x.shape[0], self.n_waypoints, 2) 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # Pre-compute positional encodings and register as buffer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # Use cached positional encodings
        return x + self.pe[:, :x.size(1)]

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,  # Reduced from 8 for better performance
        dropout: float = 0.1,
        use_checkpoint: bool = False,  # Gradient checkpointing for memory efficiency
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        
        # Optimized components
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_track * 2)
        self.encoder_embed = nn.Linear(2, d_model)
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Stack layers manually for better control and checkpointing
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ) for _ in range(num_layers)
        ])
        
        # Output projection with layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)

    def _checkpoint_layer(self, layer, tgt, memory):
        """Apply gradient checkpointing to a transformer layer"""
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(layer, tgt, memory, use_reentrant=False)
        else:
            return layer(tgt, memory)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.size(0)
        device = track_left.device
        
        # Concatenate and embed track points
        x = torch.cat([track_left, track_right], dim=1)  # (B, 2*n_track, 2)
        memory = self.encoder_embed(x)  # (B, 2*n_track, d_model)
        memory = self.pos_encoder(memory)
        
        # Create query embeddings - cache query indices for efficiency
        if not hasattr(self, '_cached_query_indices') or self._cached_query_indices.device != device:
            self._cached_query_indices = torch.arange(self.n_waypoints, device=device)
        
        tgt = self.query_embed(self._cached_query_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer layers with optional checkpointing
        for layer in self.transformer_layers:
            tgt = self._checkpoint_layer(layer, tgt, memory)
        
        # Apply final layer norm and projection
        tgt = self.layer_norm(tgt)
        output = self.output_proj(tgt)
        
        return output


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        self.cnnmodel = nn.Sequential(
            #doubling the channels each layer and reducing spatial dimension by half
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            
            nn.Flatten(),
            nn.Linear(6144, 256),  # Intermediate hidden layer
            nn.ReLU(),
            nn.Linear(256, 2 * n_waypoints)
        )
    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        output = self.cnnmodel(x)
        return output.view(x.shape[0], self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
