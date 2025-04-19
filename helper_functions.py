import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def calculate_accuracy(y_pred, y):
  top_pred = y_pred.argmax(1, keepdim=True)
  correct = top_pred.eq(y.view_as(top_pred)).sum()
  acc = correct.float() / y.shape[0]
  return acc

# Function to predict labels of images
def predict(model, dataloader, device):

  model.eval()

  images = []
  labels = []
  pred_labels = []

  with torch.inference_mode():
    for (X, y) in dataloader:

      X = X.to(device)
      y_pred = model(X)

      y_prob = F.softmax(y_pred, dim=-1)

      images.append(X.cpu())
      labels.append(y.cpu())
      pred_labels.append(torch.argmax(y_prob,1).cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)

  return images, labels, pred_labels

# Plotting n rangom images
def plot_results(examples, n_images, class_names):
    rows = int(6)
    cols = int(5)

    fig = plt.figure(figsize=(25, 15))
    for i in range(min(rows*cols, len(examples))):
        ax = fig.add_subplot(rows, cols, i+1)
        image, true_label, pred_label = examples[i]
        ax.imshow(image.permute(1,2,0), cmap='gray')
        ax.set_title(f'True: {class_names[true_label]}\n'
                     f'Predicted: {class_names[pred_label]}')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.9)
    fig.savefig('test_results.png')