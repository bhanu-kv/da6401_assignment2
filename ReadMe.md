```markdown
# DA6401 - Assignment 2: CNNs from Scratch & Transfer Learning

This repository contains two major parts:

- **Part A:** Build, train, and evaluate a custom CNN on the iNaturalist dataset
- **Part B:** Fine-tune pre-trained models using transfer learning strategies

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dataset Preparation](#dataset-preparation)
3. [Part A: Custom CNN](#part-a-custom-cnn)
4. [Part B: Transfer Learning](#part-b-transfer-learning)
5. [Requirements](#requirements)
6. [WandB Setup](#wandb-setup)
7. [Notes](#notes)

---

## Project Structure

```
project-root/
├── Part_A/                    # Custom CNN implementation
│   ├── model.py               # CNN architecture
│   ├── train_wandb.py         # Hyperparameter tuning
│   ├── test.py                # Evaluation & visualization
│   └── ...                    # Support files
├── Part_B/                    # Transfer learning
│   ├── model.py               # Pre-trained models
│   ├── train.py               # Training scripts
│   └── ...                    # Support files
└── inaturalist_12K/           # Dataset directory
```

---

## Dataset Preparation

1. Download iNaturalist 12K dataset
2. Organize directory structure:
   ```
   inaturalist_12K/
   └── inaturalist_12K/
       ├── train/
       │   └── class_folders...
       └── val/
           └── class_folders...
   ```
3. Place in project root directory

---

## Part A: Custom CNN

### Features
- Modular CNN architecture
- Hyperparameter tuning with WandB
- Bayesian optimization
- Data augmentation
- Multiple activation functions

### Usage
```
cd Part_A
python train_wandb.py -wp "myproject" -we "myname" -e 40 -b 64 -o Adam -lr 0.001
```

### Hyperparameters
| Parameter          | Values                | Default   |
|--------------------|-----------------------|-----------|
| Batch Size         | 32, 64, 128          | 64        |
| Activation         | ReLU, Mish, SiLU     | Mish      |
| Dropout            | 0.1-0.5              | 0.2       |
| Filters            | 32, 64, 128          | 64        |

---

## Part B: Transfer Learning

### Strategies
1. **Freeze Base** - Train only classifier
2. **Progressive Unfreeze** - Gradually unfreeze layers
3. **Differential LR** - Layer-specific learning rates

### Usage
```
cd Part_B
python train.py
```

---

## Requirements
```
pip install torch torchvision wandb matplotlib tqdm
```

---

## Notes
- Automatic GPU detection
- Model checkpoints saved as `.pt` files
- Adjust dataset paths in `SimpleSplitLoader` if needed
- Requires Python 3.7+

![Sample Predictions](predictions.png)
```