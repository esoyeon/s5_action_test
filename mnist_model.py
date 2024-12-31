import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import datetime
import os


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()

        # Second conv block
        self.conv2 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Third conv block
        self.conv3 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 24)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(24, 10)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool2(x)

        # Fully connected
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, epochs=1, device="cuda"):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        history = {"accuracy": [], "loss": []}
        total_batches = len(train_loader)

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            self.train()
            for i, (inputs, labels) in enumerate(train_loader):
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

                if (i + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}], Step [{i + 1}/{total_batches}], "
                        f"Loss: {running_loss/100:.4f}, "
                        f"Accuracy: {100.*correct/total:.2f}%"
                    )
                    running_loss = 0.0

            epoch_acc = correct / total
            epoch_loss = running_loss / len(train_loader)
            history["accuracy"].append(epoch_acc)
            history["loss"].append(epoch_loss)

        return history

    def save_model(self):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"mnist_model_{timestamp}.pth")

        torch.save(self.state_dict(), model_path)
        return model_path

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def load_and_preprocess_data(batch_size=32):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
