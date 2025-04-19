```markdown
# DA6401 - Assignment 2: CNNs from Scratch & Transfer Learning

This repository contains two major parts:

- **Part A:** Build, train, and evaluate a custom CNN on the iNaturalist dataset, with full hyperparameter sweep and command-line configurability.
- **Part B:** Fine-tune a large pre-trained model (e.g., ResNet50) on the same dataset using various transfer learning strategies.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Part A: Custom CNN](#part-a-custom-cnn)
  - [Features](#features)
  - [Hyperparameter Sweep](#hyperparameter-sweep)
  - [Command-Line Usage](#command-line-usage)
  - [Model Evaluation & Visualization](#model-evaluation--visualization)
- [Part B: Transfer Learning](#part-b-transfer-learning)
  - [Strategies](#strategies)
  - [Usage](#usage)
- [Requirements](#requirements)
- [WandB Setup](#wandb-setup)
- [Notes](#notes)

---

## Project Structure

```
project-root/
├── Part_A/
│   ├── model.py
│   ├── train_wandb.py
│   ├── test.py
│   ├── dataloader.py
│   ├── helper_functions.py
│   └── ...
├── Part_B/
│   ├── model.py
│   ├── train_helpers.py
│   ├── train.py
│   └── ...
├── inaturalist_12K/
│   └── inaturalist_12K/
│       ├── train/
│       └── val/
└── README.md
```

---

## Dataset Preparation

1. Download the iNaturalist 12K dataset.
2. Organize it as:
    ```
    inaturalist_12K/
      └── inaturalist_12K/
          ├── train/
          └── val/
    ```
3. Place the `inaturalist_12K` directory in the project root.

---

## Part A: Custom CNN

### Features

- Modular CNN architecture with configurable layers, filters, activation, dropout, batchnorm, pooling, and more.
- Hyperparameter tuning via [Weights & Biases (wandb)](https://wandb.ai).
- Training, validation, and test metrics tracked and visualized.
- Command-line argument parsing for all major hyperparameters.

### Hyperparameter Sweep

- **Bayesian optimization** of:
  - Number of layers, filters, kernel size, stride, padding, dropout, activation, pooling, batchnorm, augmentation, filter organization, etc.
- Early stopping with Hyperband.

### Command-Line Usage

From the `Part_A` directory, you can run:

```
python train_wandb.py \
    -wp "myprojectname" \
    -we "myname" \
    -e 40 \
    -b 64 \
    -o Adam \
    -lr 0.001 \
    -nhl 5 \
    -nhf 64 \
    -ks 3 \
    -sn 512 \
    -a "Mish" \
    -do 0.2 \
    -p "Max" \
    -bn True \
    -s 3 \
    -pd 3 \
    -fo "inc"
```

**Argument Reference:**

| Argument | Description | Default | Choices/Type |
|----------|-------------|---------|--------------|
| `-wp`, `--wandb_project` | WandB project name | "myprojectname" | string |
| `-we`, `--wandb_entity` | WandB entity name | "myname" | string |
| `-e`, `--epochs` | Number of epochs | 1 | int |
| `-b`, `--batch_size` | Batch size | 4 | int |
| `-o`, `--optimizer` | Optimizer | "sgd" | ["Adam"] |
| `-lr`, `--learning_rate` | Learning rate | 0.1 | float |
| `-nhl`, `--num_layers` | Number of CNN layers | 5 | int |
| `-nhf`, `--num_filters` | Initial filters | 64 | int |
| `-ks`, `--kernel_size` | Kernel size | 3 | int |
| `-sn`, `--hidden_neurons` | Dense layer neurons | 512 | int |
| `-a`, `--activation` | Activation | "Mish" | see code |
| `-do`, `--dropout` | Dropout | 0.2 | float |
| `-p`, `--pool` | Pooling | "Max" | ["Max", "Avg"] |
| `-bn`, `--batchnorm` | BatchNorm | False | bool |
| `-s`, `--stride` | Stride | 2 | int |
| `-pd`, `--padding` | Padding | 2 | int |
| `-fo`, `--fil_org` | Filter org | 2 | ["inc", "dec", "const"] |

### Model Evaluation & Visualization

After training, evaluate and visualize predictions:

```
python test.py
```

- Prints train, validation, and test loss/accuracy.
- Saves and displays prediction images with true and predicted labels.

---

## Part B: Transfer Learning

### Strategies

- **freeze_base:** Freeze all layers except the last (classifier).
- **progressive_unfreeze:** Gradually unfreeze deeper layers during training.
- **differential_lr:** Use different learning rates for different layers.

### Usage

From the `Part_B` directory:

```
python train.py
```

- By default, uses `freeze_base` strategy.
- To use another strategy, edit the line:
  ```
  model, optimizer = freeze_base()  # or progressive_unfreeze(), differential_lr()
  ```

- All training, validation, and test metrics are logged to wandb.

---

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- wandb
- matplotlib
- tqdm

Install dependencies:
```
pip install torch torchvision wandb matplotlib tqdm
```

---

## WandB Setup

1. [Sign up for wandb](https://wandb.ai) if you don't have an account.
2. Login in your terminal:
    ```
    wandb login
    ```

---

## Notes

- All scripts automatically use GPU if available.
- Model checkpoints are saved as `parta.pt` (Part A) and `partb.pt` (Part B).
- Adjust dataset paths in `SimpleSplitLoader` if your directory structure differs.
- For custom experiments, modify or extend the sweep configuration or model architecture as needed.

---

---
```