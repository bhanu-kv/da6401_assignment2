import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import matplotlib.pyplot as plt
import torch
from torch import nn
from model import CNN_Model
from dataloader import SimpleSplitLoader
from tqdm.auto import tqdm
import os
from train_helpers import test_step, train_step
from helper_functions import predict, plot_results

def test_wandb(config = None):
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

    conv_layers_config = []

    conv_layers_config.append({
            'in_channels': 3,
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 3,
            'padding': 3,
            'activation': 'Mish',
            'pooling': {'type': 'Max', 'kernel_size': 1, 'stride': 1}
        })

    for i in range(4):
        in_channels = 64
        out_channels = 64

        conv_layers_config.append({
            'in_channels': int(in_channels),
            'out_channels': int(out_channels),
            'kernel_size': 3,
            'stride': 3,
            'padding': 3,
            'activation': 'Mish',
            'pooling': {'type': 'Max', 'kernel_size': 1, 'stride': 1}
        })

    fc_layers_config = [
        {'out_features': 512, 'activation': 'Mish'},
    ]

    model = CNN_Model(
        conv_layers_config=conv_layers_config,
        fc_layers_config=fc_layers_config,
        input_shape=(3, 224, 224),
        batch_norm=True,
        conv_dropout=0.2,
        fc_dropout=0.2,
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

    for epoch in tqdm(range(30)):

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

    model.load_state_dict(torch.load('parta.pt'))

    print("Training Data Loss and Accuracy:")

    train_loss, train_acc = test_step(dataloader=train_loader,
                                    model=model,
                                    loss_fn=cross_en_loss,
                                    best_valid_loss = best_valid_loss,
                                    device=device,
                                    testing =True,
                                    train_data=True
                                )

    print()
    print("Validation Data Loss and Accuracy:")

    train_loss, train_acc = test_step(dataloader=val_loader,
                                    model=model,
                                    loss_fn=cross_en_loss,
                                    best_valid_loss = best_valid_loss,
                                    device=device,
                                    testing =True,
                                    valid_data=True
                                )

    print()
    print("Testing Data Loss and Accuracy:")

    train_loss, train_acc = test_step(dataloader=test_loader,
                                    model=model,
                                    loss_fn=cross_en_loss,
                                    best_valid_loss = best_valid_loss,
                                    device=device,
                                    testing =True
                                )
    
    N_IMAGES = 30
    images, labels, pred_labels = predict(model, test_loader, device)

    examples = []

    for image, label, pred_label in zip(images, labels, pred_labels):
        examples.append((image, label, pred_label))

        plot_results(examples, N_IMAGES, class_names=loader.class_names)

project_name = 'DA6401 - Assignment2'
entity = 'CE21B031'

def main():
    test_wandb()

if __name__ == "__main__":
    main()