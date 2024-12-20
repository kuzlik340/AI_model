from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import SGD
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torchvision.datasets import MNIST
import yaml
from torchvision import transforms

class CTDataset(Dataset):
    def __init__(self, filepath):
        # Creating tensor x and tensor y
        # x will contain N (60000 for train and 10000 for test)
        # matrices with size 28*28 pixels
        # y will contain real labels for images
        self.x, self.y = torch.load(filepath)
        # Normalizing tensor x so it will have interval of each pixel from 0 to 1
        self.x = self.x / 255
        # For labels creating vectors with 10 classes (from 0 to 9)
        self.y = F.one_hot(self.y, num_classes=10).float()
    def __len__(self):
        # Will return the N number of all images
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # initializing 3 layers with input of 28*28 and output as 10
        self.Matrix1 = nn.Linear(28**2, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, 10)
        self.R = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()

def train_model(dl, f,  num_epochs, learning_rate, optimizer_type, momentum = 0.9):
    if optimizer_type == 'SGD':
        opt = SGD(f.parameters(), learning_rate)
    elif optimizer_type == 'SGD_momentum':
        opt = SGD(f.parameters(), learning_rate, momentum)
    elif optimizer_type == 'Adam':
        opt = torch.optim.Adam(f.parameters(), learning_rate)
    else:
        print("Invalid optimizer type")
        exit(1)
    L = nn.CrossEntropyLoss()

    losses = []
    epochs = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}')
        N = len(dl)
        for i, (x,y) in enumerate(dl):
            opt.zero_grad()
            loss_value = L(f(x), y)
            loss_value.backward()
            opt.step()
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)

def evaluate_model(dl, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(y, 1)
            correct += (predicted == labels).sum().item()
            total += y.size(0)
    return 100 * correct / total

def plot_loss(epochs, losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label="Cross-Entropy Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Cross-Entropy Loss vs. Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def setup():
    processed_data_path = './data/processed'
    os.makedirs(processed_data_path, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    try:
        train_dataset = MNIST(root='./data', train=True, transform=transform, download=False)
        test_dataset = MNIST(root='./data', train=False, transform=transform, download=False)
        print("MNIST data found locally.")
    except RuntimeError:
        print("MNIST data not found. Downloading...")
        train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

    train_data = (train_dataset.data.float(), train_dataset.targets)
    test_data = (test_dataset.data.float(), test_dataset.targets)

    torch.save(train_data, os.path.join(processed_data_path, 'train.pt'))
    torch.save(test_data, os.path.join(processed_data_path, 'test.pt'))

    print("All data is processed and ready to be used in training and test!")


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    batch_size = config["batch_size"]
    momentum = config["momentum"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    optimizer_type = config["optimizer"]
    train_file = config["train_file"]
    test_file = config["test_file"]
    if train_file == "":
        setup()
        train_dataset = CTDataset('./data/processed/train.pt')
        test_dataset = CTDataset('./data/processed/test.pt')
    else:
        train_dataset = CTDataset(train_file)
        test_dataset = CTDataset(test_file)
    print(len(train_dataset), len(test_dataset))
    train_dl= DataLoader(train_dataset, batch_size)
    f = NeuralNetwork()
    if momentum != 0:
       epoch, losses = train_model(train_dl, f, num_epochs, learning_rate, optimizer_type, momentum)
    else:
       epoch, losses = train_model(train_dl, f, num_epochs, learning_rate, optimizer_type, momentum)
    torch.save(f.state_dict(), 'mnist_model.pth')
    print("Model is saved!")
    model = NeuralNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    test_dl = DataLoader(test_dataset, batch_size)
    accuracy = evaluate_model(test_dl, model)
    print(f"Test Accuracy: {accuracy:.2f}%")
    epoch_data_avg = epoch.reshape(20, -1).mean(axis=1)
    loss_data_avg = losses.reshape(20, -1).mean(axis=1)
    plot_loss(epoch_data_avg, loss_data_avg)

if __name__ == '__main__':
    main()

