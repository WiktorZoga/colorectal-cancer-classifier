import torch
import torch.nn as nn
import torchvision.models as models

class KMeansBaseline(nn.Module):
    """
    K-Means Clustering Baseline.
    Maps input vector spatial points to 9 learnable cluster centroids.
    Returns negative squared Euclidean distances to act as pseudo-logits for CrossEntropyLoss.
    """
    def __init__(self, num_classes: int = 9, input_dim: int = 3 * 28 * 28):
        super().__init__()
        self.num_classes = num_classes
        # Trainable centroids mapping directly to target classes
        self.centroids = nn.Parameter(torch.randn(num_classes, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten batch: [Batch, 3, 28, 28] -> [Batch, 3 * 28 * 28]
        x = x.view(x.size(0), -1) 
        
        x_norm = torch.sum(x**2, dim=1, keepdim=True)  # [Batch, 1]
        c_norm = torch.sum(self.centroids**2, dim=1, keepdim=False)  # [num_classes]
        distances = x_norm - 2 * torch.matmul(x, self.centroids.t()) + c_norm
        
        # Closer to centroid -> smaller distance -> larger negative value (higher probability logit)
        return -distances


class MedMNISTBaselineCNN(nn.Module):
    """
    Lightweight 2-layer Convolutional Neural Network designed for 28x28 MedMNIST frames.
    """
    def __init__(self, num_classes: int = 9):
        super(MedMNISTBaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 28x28 -> MaxPool -> 14x14
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 2: 14x14 -> MaxPool -> 7x7
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to [Batch, 1568]
        x = self.classifier(x)
        return x


def get_resnet18_baseline(num_classes: int = 9) -> nn.Module:
    """
    Standard ResNet-18 modified for 28x28 inputs.
    Replaces the initial large 7x7 convolution with a 3x3 layer and removes maxpooling 
    to preserve spatial details of smaller resolution medical matrices.
    """
    model = models.resnet18(weights=None)
    
    # Adapt initial layer for small resolutions (keeps 28x28 instead of heavily downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # Remove default heavy downsampling pool
    
    # Match final projection dimension to target tissue classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(model_name: str, num_classes: int = 9) -> nn.Module:
 
    model_name = model_name.lower().strip()
    
    if model_name == "kmeans":
        print("-> Initializing Classical Baseline: K-Means")
        return KMeansBaseline(num_classes=num_classes)
        
    elif model_name == "cnn_baseline":
        print("-> Initializing Shallow Deep Learning Baseline: MedMNISTBaselineCNN")
        return MedMNISTBaselineCNN(num_classes=num_classes)
        
    elif model_name in ["resnet18", "resnet"]:
        print("-> Initializing Standard Deep Learning Baseline: ResNet-18 (Adapted for 28x28)")
        return get_resnet18_baseline(num_classes=num_classes)
        
    else:
        raise ValueError(
            f"Unknown model architecture descriptor: '{model_name}'. "
            f"Supported paradigms: 'kmeans', 'cnn_baseline', 'resnet18'"
        )