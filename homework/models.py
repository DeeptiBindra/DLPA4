from pathlib import Path

import torch
import torch.nn as nn
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
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,  # Increased model dimension
        nhead: int = 8,
        num_layers: int = 6,  # Reduced layers for better generalization
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        
        # Enhanced input processing
        self.track_embed = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Separate embeddings for left and right tracks
        self.track_type_embed = nn.Embedding(2, d_model)  # 0 for left, 1 for right
        
        # Enhanced positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Transformer decoder with improved configuration
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',  # Better activation function
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer, 
            num_layers=num_layers
        )
        
        # Enhanced output projection with residual connection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        # Learnable output scaling for better coordinate prediction
        self.output_scale = nn.Parameter(torch.ones(2))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

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
        
        # Process track boundaries separately
        left_features = self.track_embed(track_left)  # (b, n_track, d_model)
        right_features = self.track_embed(track_right)  # (b, n_track, d_model)
        
        # Add track type embeddings
        left_type = self.track_type_embed(torch.zeros(batch_size, self.n_track, device=device, dtype=torch.long))
        right_type = self.track_type_embed(torch.ones(batch_size, self.n_track, device=device, dtype=torch.long))
        
        left_features = left_features + left_type
        right_features = right_features + right_type
        
        # Combine left and right track features
        track_features = torch.cat([left_features, right_features], dim=1)  # (b, 2*n_track, d_model)
        
        # Add positional encoding
        track_features = self.pos_encoder(track_features)
        
        # Create query embeddings for waypoints
        query_indices = torch.arange(self.n_waypoints, device=device)
        queries = self.query_embed(query_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer decoder
        output_features = self.transformer_decoder(queries, track_features)
        
        # Project to waypoint coordinates
        waypoints = self.output_proj(output_features)
        
        # Apply learnable scaling
        waypoints = waypoints * self.output_scale.unsqueeze(0).unsqueeze(0)
        
        return waypoints


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
