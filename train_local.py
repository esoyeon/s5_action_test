import torch
from mnist_model import MNISTModel, load_and_preprocess_data, set_seed


def main():
    # 시드 설정
    set_seed(42)

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로드
    train_loader, test_loader = load_and_preprocess_data(batch_size=32)
    print("Data loaded successfully")

    # 모델 초기화
    model = MNISTModel()
    print(f"Total Trainable Parameters: {model.get_parameter_count():,}")

    # 학습
    print("\nStarting training...")
    history = model.train_model(train_loader, epochs=1, device=device)
    print(f"Final Training Accuracy: {history['accuracy'][-1]*100:.2f}%")

    # 모델 저장
    saved_path = model.save_model()
    print(f"\nModel saved to: {saved_path}")


if __name__ == "__main__":
    main()
