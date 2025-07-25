"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn as nn

from .models import load_model, save_model
from .datasets.road_dataset import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False, batch_size=batch_size, num_workers=2)

    loss_fn = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        for batch in train_data:
            track_left = batch["track_left"].to(device)    
            track_right = batch["track_right"].to(device)   
            waypoints = batch["waypoints"].to(device)   
            waypoints_mask = batch["waypoints_mask"].to(device) 

            optimizer.zero_grad()
            waypoints_pred = model(track_left=track_left, track_right=track_right)
           
            loss = loss_fn(waypoints_pred, waypoints)  # shape: (B, n_waypoints, 2)
            loss = (loss * waypoints_mask.unsqueeze(-1)).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.add_scalar("mlpplanner_train_loss", train_loss / len(train_data), epoch)

        
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for batch in val_data:
                
                track_left = batch["track_left"].to(device)    
                track_right = batch["track_right"].to(device)   
                waypoints = batch["waypoints"].to(device)   
                waypoints_mask = batch["waypoints_mask"].to(device)
                waypoints_pred = model(track_left=track_left, track_right=track_right)
                loss = loss_fn(waypoints_pred, waypoints)
                loss = (loss * waypoints_mask.unsqueeze(-1)).mean()
                val_loss += loss.item()
        logger.add_scalar("mlpplanner_val_loss", val_loss / len(val_data), epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={train_loss / len(train_data):.4f} "
                f"val_loss={val_loss / len(val_data):.4f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    # Add more model-specific arguments as needed

    train(**vars(parser.parse_args()))
