import torch
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


def train_and_evaluate():
    set_seed()  # 재현성을 위한 시드 설정

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로드
    train_loader, test_loader = load_and_preprocess_data(batch_size=64)
    print("Data loaded successfully")

    # 모델 초기화
    model = MNISTModel()

    # 모델 파라미터 출력
    param_count = count_parameters(model)
    print(f"Parameter check: {'PASSED' if param_count < 25000 else 'FAILED'}")

    # 모델 학습
    print("\nStarting training...")
    history = model.train_model(train_loader, epochs=1, device=device)

    # 학습 결과 출력
    final_accuracy = history["accuracy"][-1]
    final_loss = history["loss"][-1]
    print(f"\nFinal training accuracy: {final_accuracy*100:.2f}%")
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Accuracy check: {'PASSED' if final_accuracy >= 0.95 else 'FAILED'}")

    # 모델 저장
    model_path = model.save_model()
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_and_evaluate()
