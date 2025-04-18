from loss import *
import wandb
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import wandb
from nn import *
import argparse

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
    wandb.log({"val_acc": acc, "val_loss": loss})
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
    
    name = f'hl_{config.hidden_layers}_bs_{config.batch_size}_acf_{config.activation_func}_lr_{config.learning_rate}_opt_{config.optimizer}_w_init_{config.weight_init}_wdecay_{config.weight_decay}'
    wandb.run.name = name
    wandb.run.save()

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

    wandb.finish()

project_name = 'DA6401 - Assignment2'
entity = 'CE21B031'

def main():
    parser = argparse.ArgumentParser(description="Train a Neural Network with WandB Sweeps")
    parser.add_argument("-wp", "--wandb_project", default="myprojectname", help="Project name for WandB")
    parser.add_argument("-we", "--wandb_entity", default="myname", help="WandB entity")
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="random")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid")
    
    args = parser.parse_args()
    
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': vars(args)
    }
    
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    wandb.agent(sweep_id, function=train_wandb)

if __name__ == "__main__":
    main()