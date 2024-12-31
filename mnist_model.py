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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 28x28x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14x16
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 14x14x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7x32
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 32, 10)  # Fully Connected Layer
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader, epochs=1, device="cuda"):
        set_seed()  # 재현성을 위한 시드 설정

        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.002, weight_decay=1e-5)

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
                        f"Epoch {epoch+1} - Batch [{i + 1}/{total_batches}], "
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


def load_and_preprocess_data(batch_size=64):
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
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
