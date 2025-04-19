import torch
from torch import nn
from Part_A.model import CNN_Model
from dataloader import SimpleSplitLoader
from helper_functions import calculate_accuracy
from tqdm.auto import tqdm

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
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f"Validation loss: {loss:.5f} | Validation accuracy: {acc:.2f}%\n")

    return loss, acc

# Example of how to use the class:
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conv_layers_config = [
        {
            'in_channels': 3,
            'out_channels': 32,
            'kernel_size': 3,
            'stride': 2,
            'padding': 2,
            'activation': 'ReLU',
            'pooling': {'type': 'Max', 'kernel_size': 2, 'stride': 2}
        },
        {
            'in_channels': 32,
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 2,
            'padding': 2,
            'activation': 'ReLU',
            'pooling': {'type': 'Max', 'kernel_size': 2, 'stride': 2}
        },
        {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 3,
            'stride': 2,
            'padding': 2,
            'activation': 'ReLU',
            'pooling': {'type': 'Max', 'kernel_size': 2, 'stride': 2}
        },
        {
            'in_channels': 128,
            'out_channels': 256,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'activation': 'ReLU',
            'pooling': {'type': 'Max', 'kernel_size': 2, 'stride': 2}
        },
                {
            'in_channels': 256,
            'out_channels': 512,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'activation': 'ReLU',
            'pooling': {'type': 'Max', 'kernel_size': 2, 'stride': 2}
        }
    ]
    fc_layers_config = [
        {'out_features': 512, 'activation': 'ReLU'},
    ]


    model = CNN_Model(
        conv_layers_config=conv_layers_config,
        fc_layers_config=fc_layers_config,
        input_shape=(3, 224, 224),
        batch_norm=True,
        conv_dropout=0.1,
        fc_dropout=0.1,
        output_classes=10,
        use_softmax=False  # Use CrossEntropyLoss
    ).to(device)

    loader = SimpleSplitLoader(
        train_dir="./inaturalist_12K/inaturalist_12K/train",
        test_dir="./inaturalist_12K/inaturalist_12K/val",
        image_size=(224, 224),
        batch_size=64,
        val_ratio=0.2,
        augment=True,
        num_workers=2
    )
    train_loader, val_loader, test_loader = loader.get_loaders()

    print(model)

    cross_en_loss = nn.CrossEntropyLoss().to(device)
    Adam = torch.optim.Adam(params=model.parameters(), lr=0.001)


    train_loss_prg = []
    valid_loss_prg = []

    train_acc_prg = []
    valid_acc_prg = []

    best_valid_loss = float('inf')

    epochs = 15

    for epoch in tqdm(range(epochs)):

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
