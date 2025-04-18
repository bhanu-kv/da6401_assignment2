import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class SimpleSplitLoader:
    def __init__(
        self,
        train_dir,
        test_dir,
        image_size=(224, 224),
        batch_size=32,
        val_ratio=0.2,
        augment=True,
        num_workers=2,
        seed=42
    ):
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) if augment else transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load full train dataset (no transform yet)
        full_train = ImageFolder(train_dir)

        # Split into train/val
        n_total = len(full_train)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        train_set, val_set = random_split(
            full_train, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

        # Helper for applying transforms to subsets
        class TransformedSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            def __getitem__(self, idx):
                x, y = self.subset[idx]
                return self.transform(x), y
            def __len__(self):
                return len(self.subset)

        self.train_dataset = TransformedSubset(train_set, self.train_transform)
        self.val_dataset = TransformedSubset(val_set, self.eval_transform)
        self.test_dataset = ImageFolder(test_dir, transform=self.eval_transform)

        # DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# loader = SimpleSplitLoader(
#         train_dir="./inaturalist_12K/inaturalist_12K/train",
#         test_dir="./inaturalist_12K/inaturalist_12K/val",
#         image_size=(224, 224),
#         batch_size=32,
#         val_ratio=0.2,
#         augment=True,
#         num_workers=2
#     )
# train_loader, val_loader, test_loader = loader.get_loaders()


# # Get a batch of training data
# inputs, classes = next(iter(train_loader))
# print(classes)
# out = torchvision.utils.make_grid(inputs[:4])
# imshow(out, title="Sample Training Images")
# plt.show()
