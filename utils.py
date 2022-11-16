import torch


def accuracy(y_hat, y):
    assert y_hat.shape == y.shape
    n_classes = y_hat.shape[-1]
    y_hat = y_hat.view(-1, n_classes)
    y = y.view(-1, n_classes)
    n_correct = torch.sum(y_hat == y)
    return n_correct / y_hat.numel()
