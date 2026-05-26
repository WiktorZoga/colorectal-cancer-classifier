import os
import sys
import yaml
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import get_dataloaders
from src.models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models on PathMNIST Test Dataset")
    
    # Pathway 1: Automated tracking via Run Directory
    parser.add_argument("--run_path", type=str, default=None,
                        help="Path to the specific run directory (e.g., runs/cnn_baseline/run_20260526_120000)")
    
    # Pathway 2: Explicit manual assignment
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit model architecture name ('kmeans', 'cnn_baseline', 'resnet18')")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Explicit path to saved model weights checkpoint .pth file")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Explicit path to the configuration yaml file")
                        
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Launching test evaluation pipeline on device: {device}")
    
    if args.run_path is not None:
        # Run Mode
        run_dir = Path(args.run_path)
        if not run_dir.exists():
            raise FileNotFoundError(f"Provided run directory does not exist: {run_dir}")
            
        model_path = run_dir / "best_model.pth"
        config_path = run_dir / "config.yaml"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Missing weights checkpoint ('best_model.pth') in: {run_dir}")
        if not config_path.exists():
            print(f"[WARNING] config.yaml not found in run directory. Falling back to global fallback configuration.")
            SCRIPT_DIR = Path(__file__).resolve().parent
            config_path = SCRIPT_DIR.parent / "config" / "config.yaml"
            
        model_name = run_dir.parent.name
        output_dir = run_dir
        
    else:
        # Manual Mode
        if not args.model or not args.weights_path:
            raise ValueError("Evaluation requires either a valid '--run_path' OR both explicit '--model' and '--weights_path' arguments.")
            
        model_name = args.model
        model_path = Path(args.weights_path)
        
        if args.config_path:
            config_path = Path(args.config_path)
        else:
            SCRIPT_DIR = Path(__file__).resolve().parent
            config_path = SCRIPT_DIR.parent / "config" / "config.yaml"
            
        if not model_path.exists():
            raise FileNotFoundError(f"Explicit weights path file target invalid: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Explicit config path file target invalid: {config_path}")
            
        output_dir = model_path.parent

    print(f"Loading configuration metrics from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Target architecture selected: '{model_name}'")
    print(f"Loading weights matrix parameters from: {model_path}")
        
    _, _, test_loader = get_dataloaders(config_path)
    
    model = get_model(model_name=model_name, num_classes=config["training"]["num_classes"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_probs = []
    
    print("Processing evaluation sequence over test distribution...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            probs = F.softmax(outputs, dim=1)
            
            # MedMNIST labels are 2D arrays [[idx]], flattening to 1D array
            all_labels.extend(labels.squeeze().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    predictions = np.argmax(all_probs, axis=1)
    
    # Compute performance metrics
    acc = accuracy_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    
    print("\nFINAL BENCHMARK PERFORMANCE")
    print(f"Test Accuracy (ACC): {acc * 100:.2f}%")
    print(f"Test Area Under ROC (AUC): {auc:.4f}\n")
    
    # Save validation metrics to local JSON payload file
    metrics = {
        "test_accuracy": float(acc),
        "test_auc": float(auc),
        "model_architecture": model_name,
        "weights_evaluated": str(model_path),
        "config_evaluated": str(config_path)
    }
    
    metrics_save_path = output_dir / "evaluation_results.json"
    with open(metrics_save_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4)
        
    print(f"Evaluation telemetry logs generated and saved directly to: {metrics_save_path}")