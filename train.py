import torch
import torch.nn as nn
import torch.optim as optim
from mnist_model import MNISTModel, load_and_preprocess_data, set_seed


def count_parameters(model):
    print("\nDetailed Model Parameters:")
    print("-" * 80)
    print(f"{'Layer':<40} {'Output Shape':<20} {'Param #'}")
    print("-" * 80)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
        shape = list(parameter.shape)
        print(f"{name:<40} {str(shape):<20} {param:,}")

    print("-" * 80)
    print(f"Total Trainable Parameters: {total_params:,}")
    return total_params


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_loader)

    set_seed()  # 재현성을 위한 시드 설정

    print(f"\nEpoch 1/{1}")
    print("-" * 60)

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch 1 - Batch [{i + 1}/{total_batches}], "
                f"Loss: {running_loss/100:.4f}, "
                f"Accuracy: {100.*correct/total:.2f}%"
            )
            running_loss = 0.0

    return 100.0 * correct / total


def main():
    set_seed()  # 재현성을 위한 시드 설정

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로드
    train_loader, _ = load_and_preprocess_data(batch_size=64)
    print("Data loaded successfully")

    # 모델 초기화
    model = MNISTModel().to(device)

    # 모델 파라미터 출력
    count_parameters(model)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

    # 모델 학습
    print("\nStarting training...")
    accuracy = train_model(model, train_loader, criterion, optimizer, device)
    print(f"Final training accuracy: {accuracy:.2f}%")

    # 모델 저장
    model_path = model.save_model()
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
