from helper_functions import calculate_accuracy
from tqdm.auto import tqdm
import wandb
import torch

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
    # wandb.log({"train_acc": train_acc, "train_loss": train_loss})
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
            torch.save(model.state_dict(), 'parta.pt')

        print(f"Validation loss: {loss:.5f} | Validation accuracy: {acc:.2f}%\n")

    # wandb.log({"val_acc": acc, "val_loss": loss})

    return loss, acc