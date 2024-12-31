import torch
import pytest
import time
from mnist_model import MNISTModel, load_and_preprocess_data


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    model = MNISTModel()
    model.to(device)
    return model


@pytest.fixture
def dataloaders():
    return load_and_preprocess_data()


def test_parameter_count(model):
    param_count = model.get_parameter_count()
    assert (
        param_count < 25000
    ), f"Model has {param_count} parameters, should be less than 25000"


def test_model_accuracy(model, dataloaders, device):
    train_loader, _ = dataloaders
    # batch_size를 64에서 32로 변경
    subset_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_loader.dataset, range(1000)),
        batch_size=32,  # 64 -> 32
    )
    history = model.train_model(subset_loader, epochs=1, device=device)
    final_accuracy = history["accuracy"][-1]
    assert (
        final_accuracy >= 0.95
    ), f"Model accuracy {final_accuracy} is less than required 0.95"


def test_input_output_shape(model, device):
    test_input = torch.randn(1, 1, 28, 28).to(device)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape {output.shape} is incorrect"


def test_model_runtime(model, dataloaders, device):
    train_loader, _ = dataloaders
    subset_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_loader.dataset, range(100)),
        batch_size=32,  # 이미 32로 되어있음
    )
    start_time = time.time()
    model.train_model(subset_loader, epochs=1, device=device)
    training_time = time.time() - start_time
    assert (
        training_time < 30
    ), f"Training took {training_time} seconds, should be under 30 seconds"


def test_model_structure(model):
    expected_layers = [
        "Conv2d",
        "BatchNorm2d",
        "ReLU",
        "Conv2d",
        "BatchNorm2d",
        "ReLU",
        "MaxPool2d",
        "Conv2d",
        "BatchNorm2d",
        "ReLU",
        "MaxPool2d",
        "Linear",
        "ReLU",
        "Linear",
    ]
    actual_layers = [
        layer.__class__.__name__
        for name, layer in model.named_modules()
        if not isinstance(layer, MNISTModel)
    ]
    assert (
        actual_layers == expected_layers
    ), "Model structure doesn't match expected architecture"


def test_prediction_range(model, device):
    test_input = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        predictions = torch.softmax(model(test_input), dim=1)
    assert torch.all(predictions >= 0) and torch.all(
        predictions <= 1
    ), "Predictions should be probabilities between 0 and 1"
