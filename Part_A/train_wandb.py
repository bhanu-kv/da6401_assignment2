import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import wandb
import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn
from Part_A.model import CNN_Model
from dataloader import SimpleSplitLoader
from helper_functions import calculate_accuracy
from tqdm.auto import tqdm
import os
from train_helpers import test_step, train_step


def train_wandb(config = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    run = wandb.init(config=config, resume="allow")
    config = wandb.config
    
    name = f'fl_{config.num_filters}_acf_{config.activation}_aug_{config.augmentation}_norm_{config.batchnorm}_dropout_{config.dropout}_org_{config.fil_org}'
    wandb.run.name = name
    wandb.run.save()
    conv_layers_config = []
    
    in_channels = config.num_filters
    out_channels = config.num_filters

    conv_layers_config.append({
            'in_channels': 3,
            'out_channels': config.num_filters,
            'kernel_size': config.kernel_size,
            'stride': config.stride,
            'padding': config.padding,
            'activation': config.activation,
            'pooling': {'type': config.pool, 'kernel_size': 1, 'stride': 1}
        })

    for i in range(config.num_layers-1):
        in_channels = out_channels

        if config.fil_org == 'inc': out_channels*=2
        elif config.fil_org == 'dec': out_channels/=2

        conv_layers_config.append({
            'in_channels': int(in_channels),
            'out_channels': int(out_channels),
            'kernel_size': config.kernel_size,
            'stride': config.stride,
            'padding': config.padding,
            'activation': config.activation,
            'pooling': {'type': config.pool, 'kernel_size': 1, 'stride': 1}
        })

    fc_layers_config = [
        {'out_features': config.hidden_neurons, 'activation': config.activation},
    ]

    model = CNN_Model(
        conv_layers_config=conv_layers_config,
        fc_layers_config=fc_layers_config,
        input_shape=(3, 224, 224),
        batch_norm=config.batchnorm,
        conv_dropout=config.dropout,
        fc_dropout=config.dropout,
        output_classes=10,
        use_softmax=False  # Use CrossEntropyLoss
    ).to(device)

    cross_en_loss = nn.CrossEntropyLoss().to(device)
    Adam = torch.optim.Adam(params=model.parameters(), lr=0.001)


    train_loss_prg = []
    valid_loss_prg = []

    train_acc_prg = []
    valid_acc_prg = []

    best_valid_loss = float('inf')

    for epoch in tqdm(range(config.epochs)):

        print(f"Epoch: {epoch+1}\n---------")

        train_loss, train_acc = train_step(train_loader=train_loader,
                model=model,
                loss_fn=cross_en_loss,
                device=device,
                optimizer=Adam
        )
        train_loss_prg.append(train_loss.detach().cpu().numpy())
        train_acc_prg.append(train_acc.detach().cpu().numpy())

        valid_loss, valid_acc = test_step(dataloader=val_loader,
                model=model,
                loss_fn=cross_en_loss,
                best_valid_loss = best_valid_loss,
                device=device,
                testing = False
        )
        valid_loss_prg.append(valid_loss)
        valid_acc_prg.append(valid_acc)
    
    os.remove("./parta.pt")
    wandb.finish()

project_name = 'DA6401 - Assignment2'
entity = 'CE21B031'

def main():
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'batch_size': {'values': [32]},
            'num_layers': {'values': [5]},
            'num_filters': {'values': [64]},
            'kernel_size': {'values': [3]},
            'epochs': {'values': [40]},
            'stride': {'values': [3]},
            'padding': {'values': [3]},
            'dropout': {'values': [0.2]},
            'fil_org': {'values': ['inc', 'const']},
            'pool': {'values': ['Max']},
            'hidden_neurons': {'values': [512]},
            'activation': {'values': ["ReLU", "Mish", "SiLU"]},
            'batchnorm': {'values': [True]},
            'augmentation': {'values': [True]}
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "max_iter": 20,
            "eta": 3
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
    wandb.agent(sweep_id, function=train_wandb)

if __name__ == "__main__":
    main()