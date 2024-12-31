import torch
from mnist_model import MNISTModel, load_and_preprocess_data
import matplotlib.pyplot as plt


def train_and_evaluate():
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로드
    train_loader, test_loader = load_and_preprocess_data(batch_size=64)
    print("Data loaded successfully")

    # 모델 초기화
    model = MNISTModel().to(device)
    print(f"Total parameters: {model.get_parameter_count():,}")

    # 모델 학습 (1 epoch)
    print("\nStarting training...")
    history = model.train_model(train_loader, epochs=1, device=device)

    # 학습 결과 출력
    final_accuracy = history["accuracy"][-1]
    final_loss = history["loss"][-1]
    print(f"\nFinal training accuracy: {final_accuracy:.4f}")
    print(f"Final training loss: {final_loss:.4f}")
    print(
        f"Required accuracy (0.95) {'achieved' if final_accuracy >= 0.95 else 'not achieved'}"
    )

    # 모델 저장
    model_path = model.save_model()
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_and_evaluate()
