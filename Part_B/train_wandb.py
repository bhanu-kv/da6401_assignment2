from train_helpers import train_model
from model import differential_lr, freeze_base, progressive_unfreeze, get_model
import sys
from pathlib import Path
import torch
import wandb

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from dataloader import SimpleSplitLoader

# Example usage with dummy dataset
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize strategy (change method number)
    model, preprocess = get_model()

    loader = SimpleSplitLoader(
        train_dir="../inaturalist_12K/inaturalist_12K/train",
        test_dir="../inaturalist_12K/inaturalist_12K/val",
        image_size=(224, 224),
        batch_size=64,
        val_ratio=0.2,
        augment=True,
        num_workers=2
    )
    train_loader, val_loader, test_loader = loader.get_loaders()
    
    # Select method
    model, optimizer = freeze_base()  # Change method here
    model.to(device)
    
    # Train
    project_name = 'DA6401 - Assignment2'
    entity = 'CE21B031'

    wandb.init(project=project_name, entity=entity)
    train_model(model, optimizer, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, model_name='freeze_base')
    wandb.finish()
    

