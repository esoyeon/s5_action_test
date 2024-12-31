import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import datetime
import os


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        # First Block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # 28x28 -> 26x26
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        # Second Block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 26x26 -> 24x24
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 24x24 -> 12x12

        # Third Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)  # 12x12 -> 10x10
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 10x10 -> 5x5

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool2(x)

        x = x.view(-1, 64 * 5 * 5)  # Flatten
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, epochs=1, device="cuda"):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters())

        history = {"accuracy": [], "loss": []}

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            self.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total

            history["accuracy"].append(epoch_acc)
            history["loss"].append(epoch_loss)

        return history

    def save_model(self):
        # Create models directory if it doesn't exist
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        # Create model path with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"mnist_model_{timestamp}.pth")

        # Save the model
        torch.save(self.state_dict(), model_path)
        return model_path

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def load_and_preprocess_data(batch_size=32):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
