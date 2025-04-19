import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet50, ResNet50_Weights

def get_model(num_classes=10):
    """Base model setup with pretrained weights"""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    # Replace and unfreeze final FC layer
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model, weights.transforms()  

# Freeze all except final layer
def freeze_base():
    model, preprocess = get_model()
    
    # Freeze base network
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze FC layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.fc.parameters(), lr=3e-4)
    return model, optimizer

# Progressive unfreezing
def progressive_unfreeze():
    model, preprocess = get_model()
    
    # Phase 1: Train only FC
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=3e-4)
    
    return model, optimizer

# Differential learning rates
def differential_lr():
    model, preprocess = get_model()
    
    # Group parameters by network depth
    params_group = [
        {'params': model.fc.parameters(), 'lr': 1e-3},      # High LR for new layers
        {'params': model.layer4.parameters(), 'lr': 1e-4},  # Medium LR
        {'params': model.layer3.parameters(), 'lr': 1e-5},  # Low LR
        {'params': model.layer2.parameters(), 'lr': 1e-6},  # Very low LR
        {'params': model.layer1.parameters(), 'lr': 1e-6},  # Frozen in practice
    ]
    
    # Freeze stem layers (conv1, bn1, etc.)
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(params_group)
    return model, optimizer