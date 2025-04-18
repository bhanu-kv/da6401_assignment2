import wandb
import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn
from model import CNN_Model
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
    wandb.log({"val_acc": train_acc, "val_loss": train_loss})
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

    wandb.log({"val_acc": acc, "val_loss": loss})

    return loss, acc
  
def train_wandb(config = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    run = wandb.init(config=config, resume="allow")
    config = wandb.config
    
    name = f'fl_{config.num_filters}_bs_{config.batch_size}_acf_{config.activation}_aug_{config.augmentation}_norm_{config.batchnorm}_dropout_{config.dropout}'
    wandb.run.name = name
    wandb.run.save()
    conv_layers_config = []

    conv_layers_config.append({
            'in_channels': 3,
            'out_channels': config.num_filters,
            'kernel_size': config.kernel_size,
            'stride': config.stride,
            'padding': config.padding,
            'activation': config.activation,
            'pooling': {'type': config.pool, 'kernel_size': 1, 'stride': 1}
        })
    
    curr_kernel_size = config.kernel_size
    mult_factor = 1
    if config.fil_org == 'inc': mult_factor = 2
    elif config.fil_org == 'dec': mult_factor = 0.5
    else: mult_factor = 1

    for i in range(config.num_layers-1):
        conv_layers_config.append({
            'in_channels': curr_kernel_size,
            'out_channels': curr_kernel_size*mult_factor,
            'kernel_size': config.kernel_size,
            'stride': config.stride,
            'padding': config.padding,
            'activation': config.activation,
            'pooling': {'type': config.pool, 'kernel_size': 1, 'stride': 1}
        })

        curr_kernel_size*mult_factor

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
            'num_filters': {'values': [32, 64, 128]},
            'kernel_size': {'values': [3]},
            'epochs': {'values': [30]},
            'stride': {'values': [2]},
            'padding': {'values': [2]},
            'dropout': {'values': [0.2, 0.3]},
            'fil_org': {'values': ['inc', 'dec', 'const']},
            'pool': {'values': ['Max']},
            'hidden_neurons': {'values': [512]},
            'activation': {'values': ["ReLU", "Mish", "GELU", "SiLU"]},
            'batchnorm': {'values': [True, False]},
            'augmentation': {'values': [True, False]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
    wandb.agent(sweep_id, function=train_wandb)

if __name__ == "__main__":
    main()