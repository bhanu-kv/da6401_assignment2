import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import torch
import torch.nn as nn
import wandb
from helper_functions import calculate_accuracy
from tqdm.auto import tqdm
from torch import optim
import os

def train_step(model, train_loader, loss_fn, optimizer, device):
    train_loss = 0
    train_acc = 0

    model.train()

    for (X, y) in tqdm(train_loader, desc="Training", leave=False):
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += calculate_accuracy(y_pred, y)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

    #Divide total train loss by length of train dataloader
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_acc*=100

    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    wandb.log({"train_acc": train_acc, "train_loss": train_loss})
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, best_valid_loss, device, testing, train_data = False, valid_data = False):
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for (X, y) in tqdm(dataloader, desc="Evaluating", leave=False, disable=True):
      X = X.to(device)
      y = y.to(device)

      # Forward pass
      eval_pred = model(X)

      # Calculate loss
      loss += loss_fn(eval_pred, y)

      # Calculate accuracy
      acc += calculate_accuracy(eval_pred, y)

    # Divide total test loss by length of test dataloader (per batch)
    loss /= len(dataloader)
    acc /= len(dataloader)
    acc*=100

    # When we want to check Final Model Loss and Accuracy
    if testing == True:
        if train_data == True:
            print(f"Train Data loss: {loss:.5f} | Train Data Accuracy: {acc:.2f}%\n")
            return loss, acc
        elif valid_data == True:
            print(f"Valid Data loss: {loss:.5f} | Valid Data Accuracy: {acc:.2f}%\n")
            return loss, acc
        else:
            print(f"Test loss: {loss:.5f} | Test accuracy: {acc:.2f}%\n")

    # While Training the Model
    elif testing == False:
        if loss < best_valid_loss:
            best_valid_loss = loss
            torch.save(model.state_dict(), 'partb.pt')

        print(f"Validation loss: {loss:.5f} | Validation accuracy: {acc:.2f}%\n")

    wandb.log({"val_acc": acc, "val_loss": loss})

    return loss, acc
  
# Generic training function
def train_model(model, optimizer, train_loader, val_loader, test_loader, epochs=15, model_name='base_freeze'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cross_en_loss = nn.CrossEntropyLoss().to(device)

    train_loss_prg = []
    valid_loss_prg = []

    train_acc_prg = []
    valid_acc_prg = []

    best_valid_loss = float('inf')

    for epoch in tqdm(range(epochs)):

        print(f"Epoch: {epoch+1}\n---------")

        if model_name == 'progressive_unfreeze' and epochs >= 5 and epochs <=10:
            optimizer = optim.Adam([
                {'params': model.fc.parameters(), 'lr': 3e-4},
                {'params': model.layer4.parameters(), 'lr': 1e-5}
            ])
        elif model_name == 'progressive_unfreeze' and epochs > 10:
            optimizer = optim.Adam([
                {'params': model.fc.parameters(), 'lr': 3e-4},
                {'params': model.layer4.parameters(), 'lr': 1e-5},
                {'params': model.layer3.parameters(), 'lr': 1e-6}
            ])

        train_loss, train_acc = train_step(train_loader=train_loader,
                model=model,
                loss_fn=cross_en_loss,
                device=device,
                optimizer=optimizer
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

    os.remove("./partb.pt")
    wandb.finish()