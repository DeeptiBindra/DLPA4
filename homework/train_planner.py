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

from .models import load_model, save_model, CNNPlanner  # <-- Added CNNPlanner import
from .datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    use_compile: bool = True,
    use_mixed_precision: bool = True,
    use_checkpoint: bool = False,  # Enable gradient checkpointing for memory efficiency
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Pass checkpointing option to transformer model
    model_kwargs = kwargs.copy()
    if model_name == "transformer_planner":
        model_kwargs['use_checkpoint'] = use_checkpoint

    model = load_model(model_name, **model_kwargs).to(device)
    
    # Apply torch.compile for performance optimization
    if use_compile and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Failed to compile model: {e}")
            use_compile = False
    
    model.train()

    # Optimized data loading with more workers and pin_memory
    num_workers = min(8, torch.get_num_threads())
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, 
                          num_workers=num_workers)
    val_data = load_data("drive_data/val", shuffle=False, batch_size=batch_size, 
                        num_workers=num_workers)

    loss_fn = nn.MSELoss(reduction="none")
    if model_name == "transformer_planner":
        loss_fn = nn.L1Loss(reduction="none")
        num_epoch = 100
    
    # Use AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    
    # Mixed precision training setup
    scaler = torch.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    
    print(f"Training {model_name} for {num_epoch} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision: {use_mixed_precision and device.type == 'cuda'}")
    print(f"Torch compile: {use_compile}")
    print(f"Gradient checkpointing: {use_checkpoint}")

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_data):
            optimizer.zero_grad()
            waypoints = batch["waypoints"].to(device, non_blocking=True)
            waypoints_mask = batch["waypoints_mask"].to(device, non_blocking=True)

            # Use mixed precision if available
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                if model_name == "cnn_planner":
                    image = batch["image"].to(device, non_blocking=True)
                    waypoints_pred = model(image=image)
                else:
                    track_left = batch["track_left"].to(device, non_blocking=True)
                    track_right = batch["track_right"].to(device, non_blocking=True)
                    waypoints_pred = model(track_left=track_left, track_right=track_right)

                loss = loss_fn(waypoints_pred, waypoints)  # shape: (B, n_waypoints, 2)
                loss = (loss * waypoints_mask.unsqueeze(-1)).mean()
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()

        # Update learning rate
        scheduler.step()
        
        logger.add_scalar(f"{model_name}_train_loss", train_loss / len(train_data), epoch)
        logger.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)

        # Validation
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for batch in val_data:
                waypoints = batch["waypoints"].to(device, non_blocking=True)
                waypoints_mask = batch["waypoints_mask"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                    if model_name == "cnn_planner":
                        image = batch["image"].to(device, non_blocking=True)
                        waypoints_pred = model(image=image)
                    else:
                        track_left = batch["track_left"].to(device, non_blocking=True)
                        track_right = batch["track_right"].to(device, non_blocking=True)
                        waypoints_pred = model(track_left=track_left, track_right=track_right)

                    loss = loss_fn(waypoints_pred, waypoints)
                    loss = (loss * waypoints_mask.unsqueeze(-1)).mean()
                    val_loss += loss.item()

        logger.add_scalar(f"{model_name}_val_loss", val_loss / len(val_data), epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={train_loss / len(train_data):.4f} "
                f"val_loss={val_loss / len(val_data):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

    # Save model without compilation for compatibility
    if use_compile and hasattr(model, '_orig_mod'):
        save_model(model._orig_mod)
        torch.save(model._orig_mod.state_dict(), log_dir / f"{model_name}.th")
    else:
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
    parser.add_argument("--use_compile", action="store_true", default=True,
                       help="Use torch.compile for optimization")
    parser.add_argument("--use_mixed_precision", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--use_checkpoint", action="store_true", default=False,
                       help="Use gradient checkpointing for memory efficiency")
    # Add more model-specific arguments as needed

    train(**vars(parser.parse_args()))
