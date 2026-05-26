import os
import sys
import yaml
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Pipeline for Colorectal Cancer Tissue Classification")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the target input image (e.g. data/sample.tif)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model architecture matching the checkpoint ('kmeans', 'cnn_baseline', 'resnet18')")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to saved model checkpoint weights (e.g. runs/cnn_baseline/run_xxx/best_model.pth)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR.parent / "config" / "config.yaml"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    img_size = int(config["data"]["image_size"])
    
    class_mapping = {
        0: "ADI (Adipose)", 
        1: "BACK (Background)", 
        2: "DEB (Debris)", 
        3: "LYM (Lymphocytes)", 
        4: "MUC (Mucus)", 
        5: "MUS (Smooth Muscle)", 
        6: "NORM (Normal Mucosa)", 
        7: "PRT (Tumor Epithelium)", 
        8: "TRU (Stroma)"
    }
    
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Target input image path invalid: {args.image_path}")
        
    img = Image.open(args.image_path).convert("RGB")
    
    inference_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Transform image and append batch dimension: [3, 28, 28] -> [1, 3, 28, 28]
    input_tensor = inference_transforms(img).unsqueeze(0).to(device)
    
    model = get_model(model_name=args.model, num_classes=config["training"]["num_classes"])
    
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Checkpoint target file path does not exist: {args.weights_path}")
        
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"\nRunning model evaluation on {device}...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
    predicted_idx = int(np.argmax(probabilities))
    confidence = probabilities[predicted_idx] * 100
    
    print("\nINFERENCE RESULTS")
    print(f"Predicted Class : {class_mapping[predicted_idx]}")
    print(f"Model Confidence: {confidence:.2f}%\n")