import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchinfo import summary

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Define transformations for the training and validation sets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match AlexNet's input size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Pre-trained model normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Path to the root directory where your images are stored
data_dir = 'dataset'

# Load the datasets from the folder structure
image_datasets = {
    'train': datasets.ImageFolder(root=f'{data_dir}/train', transform=data_transforms['train']),
    'val': datasets.ImageFolder(root=f'{data_dir}/val', transform=data_transforms['val']),
}

# Create DataLoaders for each dataset
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4),
}

# Get the class names (the folder names)
class_names = image_datasets['train'].classes
print(f"Classes: {class_names}")

# Load the pre-trained AlexNet model
#model = models.alexnet(pretrained=True)
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# Modify the final layer for transfer learning (assuming you have N classes)
num_classes = len(class_names)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
for x in model.features.parameters(): x.requires_grad = False

# Print the model to verify the architecture
summary(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Get the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Track loss and accuracy for each epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Number of epochs
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    # Initialize confusion matrix-related variables
    all_preds = []
    all_labels = []
    
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set the model to training mode
        else:
            model.eval()   # Set the model to evaluation mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over the data
        for inputs, labels in dataloaders[phase]:
            # Remove .to('cuda') to keep the data on the CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # For confusion matrix (only in the validation phase)
            if phase == 'val':
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        # Calculate epoch statistics
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Store the losses and accuracies for plotting later
        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
        else:
            val_losses.append(epoch_loss)
            val_accuracies.append(epoch_acc.item())

    # Generate confusion matrix after each validation epoch
    if phase == 'val':
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        #plt.show()
        plt.savefig(f'metrics_results/confusion_matrix-{epoch}.png')  # Save as a PNG file
        plt.close()

# Plot Loss and Accuracy Curves
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
#plt.show()
plt.savefig('metrics_results/loss_accuracy_curves.png')  # Save as a PNG file
plt.close()

torch.save(model.state_dict(), 'alex_weights.pth')