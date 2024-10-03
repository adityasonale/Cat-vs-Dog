import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import ImageLoader
from model import CNNNetwork

# Setting Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters

epochs = 10
lr = 0.001
n_output = 2



dataset_path = r"D:\Datasets\cats_dogs"

dataset = ImageFolder(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(dataset.imgs, dataset.targets, test_size=0.2, random_state=42)


train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

test_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])


train_dataset = ImageLoader(dataset=X_train,transform=train_transform)
test_dataset = ImageLoader(dataset=X_test,transform=test_transform)


# Creating data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# initialising models
model = CNNNetwork(n_outputs=n_output).to(device)

# Optimizer
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# Loss
criterion = nn.CrossEntropyLoss()


# Implementing training loop
for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs}")

            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()  # Computes the gradients of the loss with respect to the model parameters
                optimizer.step() # Updates the model’s parameters using the gradients that were computed during loss.backward()
                
                # Accumulate loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar with current loss and accuracy
                tepoch.set_postfix(loss=running_loss/total, accuracy=100*correct/total)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/total:.4f}, Accuracy: {100*correct/total:.2f}%")