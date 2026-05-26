import yaml
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

def get_dataloaders(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    db_info = INFO['pathmnist']
    DataClass = getattr(medmnist, db_info['python_class'])
    
    train_dataset = DataClass(split='train', transform=data_transform, download=True, root="datasets")
    val_dataset = DataClass(split='val', transform=data_transform, download=True, root="datasets")
    test_dataset = DataClass(split='test', transform=data_transform, download=True, root="datasets")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader