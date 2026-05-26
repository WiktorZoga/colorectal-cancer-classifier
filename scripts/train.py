import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import get_dataloaders
from src.models import get_model

def parse_args_bootstrap():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Explicit path to a baseline configuration yaml file")
    args, _ = parser.parse_known_args()
    return args

def parse_args(config, default_config_path):
    parser = argparse.ArgumentParser(description="Train MedMNIST Classifier and Baseline Models")
    
    # Configuration Path tracking
    parser.add_argument("--config_path", type=str, default=str(default_config_path),
                        help="Path to the configuration yaml file used for training parameters")
    
    # Overrides mapping dynamically from loaded config
    parser.add_argument("--model", type=str, default=config["training"]["model_name"],
                        help="Model architecture ('kmeans', 'cnn_baseline', 'resnet18')")
    parser.add_argument("--epochs", type=int, default=config["training"]["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config["training"]["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=config["data"]["batch_size"],
                        help="Batch size for training")
                        
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).squeeze().long()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)
        
    return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting training pipeline on device: {device}")
    
    bootstrap_args = parse_args_bootstrap()
    if bootstrap_args.config_path is not None:
        resolved_config_path = Path(bootstrap_args.config_path)
    else:
        resolved_config_path = PROJECT_ROOT / "config" / "config.yaml"
        
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Target configuration file path invalid: {resolved_config_path}")
        
    print(f"Loading base configuration metrics from: {resolved_config_path}")
    with open(resolved_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    args = parse_args(base_config, resolved_config_path)
    
    # Update config runtime dict entries with current CLI overrides
    base_config["data"]["batch_size"] = args.batch_size
    base_config["training"]["epochs"] = args.epochs
    base_config["training"]["learning_rate"] = args.lr
    base_config["training"]["model_name"] = args.model
    
    # Setup structured logging output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / args.model / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_path = run_dir / "best_model.pth"
    
    # Save the exact current configuration state snapshot inside the run directory
    run_config_path = run_dir / "config.yaml"
    with open(run_config_path, "w") as f:
        yaml.safe_dump(base_config, f, default_flow_style=False)
    print(f"[CONFIG] Runtime configuration saved snapshot to: {run_config_path}")
    
    # Initialize Weights & Biases
    wandb.init(
        project="colorectal-cancer-classifier",
        name=f"{args.model}_{timestamp}",
        config={
            "learning_rate": args.lr,
            "architecture": args.model, 
            "dataset": "PathMNIST",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "run_dir": str(run_dir),
            "source_config": str(resolved_config_path)
        }
    )
    
    # Save W&B metadata locally for tracking
    with open(run_dir / "wandb_info.txt", "w") as f:
        f.write(f"Run ID: {wandb.run.id}\n")
        f.write(f"Run URL: {wandb.run.get_url()}\n")

    # Instantiate architecture via factory mapping using config path file environment variables
    model = get_model(model_name=args.model, num_classes=base_config["training"]["num_classes"]).to(device)
        
    # Crucial: Pass the exact configuration file path resolved to the dataloader function
    train_loader, val_loader, _ = get_dataloaders(resolved_config_path)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f" Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% || Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"   [SAVED] New best checkpoint saved to: {model_save_path}")

    print(f"\nTraining completed. Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Artifacts stored securely in: {run_dir}")
    
    wandb.finish()