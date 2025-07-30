#!/usr/bin/env python3
"""
Performance optimization script for TransformerPlanner
This script retrains the model with all optimizations enabled.
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path

from models import load_model, save_model
from datasets.road_dataset import load_data
from train_planner import train


def benchmark_model(model, device, batch_size=32, num_iterations=100):
    """Benchmark model inference performance"""
    model.eval()
    
    # Create dummy input
    n_track = 10
    track_left = torch.randn(batch_size, n_track, 2).to(device)
    track_right = torch.randn(batch_size, n_track, 2).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(track_left=track_left, track_right=track_right)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(track_left=track_left, track_right=track_right)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    throughput = batch_size / avg_time
    
    return avg_time * 1000, throughput  # ms, samples/sec


def main():
    parser = argparse.ArgumentParser(description='Optimize TransformerPlanner performance')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model with optimizations')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark current model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.benchmark:
        print("\n=== BENCHMARKING CURRENT MODEL ===")
        try:
            # Load existing model
            model = load_model('transformer_planner', with_weights=True).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            model_size_mb = total_params * 4 / 1024 / 1024
            
            print(f"Model parameters: {total_params:,}")
            print(f"Model size: {model_size_mb:.1f} MB")
            
            # Benchmark
            avg_time, throughput = benchmark_model(model, device)
            print(f"Average inference time: {avg_time:.2f} ms")
            print(f"Throughput: {throughput:.1f} samples/sec")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    if args.retrain:
        print("\n=== RETRAINING WITH OPTIMIZATIONS ===")
        
        # Backup existing model
        model_path = Path("transformer_planner.th")
        if model_path.exists():
            backup_path = Path("transformer_planner_backup.th")
            model_path.rename(backup_path)
            print(f"Backed up existing model to {backup_path}")
        
        # Train with optimizations
        print("Training optimized TransformerPlanner...")
        train(
            model_name="transformer_planner",
            batch_size=args.batch_size,
            num_epoch=args.epochs,
            lr=args.lr,
            use_compile=True,
            use_mixed_precision=True,
            use_checkpoint=False,  # Start without checkpointing for speed
            # Model architecture optimizations
            d_model=64,
            nhead=8,
            num_layers=4,
            dropout=0.1,
        )
        
        print("\n=== BENCHMARKING OPTIMIZED MODEL ===")
        try:
            # Load newly trained model
            model = load_model('transformer_planner', with_weights=True).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            model_size_mb = total_params * 4 / 1024 / 1024
            
            print(f"Optimized model parameters: {total_params:,}")
            print(f"Optimized model size: {model_size_mb:.1f} MB")
            
            # Benchmark
            avg_time, throughput = benchmark_model(model, device)
            print(f"Optimized inference time: {avg_time:.2f} ms")
            print(f"Optimized throughput: {throughput:.1f} samples/sec")
            
        except Exception as e:
            print(f"Error benchmarking optimized model: {e}")


if __name__ == "__main__":
    main()