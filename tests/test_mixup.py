import pytest
import torch
import torch.nn as nn
from pytest_mock import MockFixture

from src.mixup import MixupCrossentropy, mixup


@pytest.fixture()
def data() -> torch.Tensor:
    data = torch.zeros((2, 3, 10, 10)).float()
    data[0] = 1.0

    return data


@pytest.fixture()
def labels() -> torch.Tensor:
    lables = torch.tensor([0, 1]).long()

    return lables


@pytest.fixture()
def preds() -> torch.Tensor:
    preds = torch.rand((2, 2))

    return preds


def test_mixup(mocker: MockFixture, data: torch.Tensor, labels: torch.Tensor) -> None:
    mocker.patch("numpy.random.beta").return_value = 0.5
    mocker.patch("torch.randperm").return_value = torch.tensor([1, 0]).long()

    mixed_data, labels1, labels2, lamb = mixup(data, labels, alpha=1.0)

    assert torch.all(labels2 == torch.tensor([1, 0]))
    assert torch.all(mixed_data == torch.ones((2, 3, 10, 10)).float() * 0.5)

    mixed_data, labels1, labels2, lamb = mixup(data, labels, alpha=0.0)

    assert torch.all(mixed_data == data)


class TestMixupCrossEntropy:
    @pytest.fixture()
    def criterion(self) -> MixupCrossentropy:
        return MixupCrossentropy()

    def test_forward(
        self,
        criterion: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
    ) -> None:
        mixed_data, labels1, labels2, lamb = mixup(data, labels, alpha=1.0)

        loss = criterion(preds, labels1, labels2, lamb)

        assert loss > 0

        with pytest.raises(ValueError):
            criterion(preds, labels1, labels2, -1)

        with pytest.raises(ValueError):
            criterion(preds, labels1, labels2, 100)
