"""Common datasets."""

import os

import numpy as np
import torch
import torchvision
from torchvision.datasets.utils import download_and_extract_archive


class IsNumber:
  """Is a digit a given number?

  Toy binary classification task on MNIST digits.
  """

  def __init__(self, n, train=True):
    assert 0 <= n <= 9, 'n must be in {0, 1, ..., 9}'
    self.n = n

    mnist = torchvision.datasets.MNIST('/tmp/mnist', download=True, train=train)
    x = mnist.data
    y = mnist.targets

    self.x = x
    self.y = torch.where(y == n, 1, 0)

  def __getitem__(self, idx):
    return self.x[idx] / 255., self.y[idx].float()

  def __len__(self):
    return len(self.x)


class MNIST:
  """MNIST dataset."""

  def __init__(self, train=True):
    mnist = torchvision.datasets.MNIST('/tmp/mnist', download=True, train=train)
    self.x = mnist.data
    self.y = mnist.targets

  def __getitem__(self, idx):
    return self.x[idx] / 255., self.y[idx]

  def __len__(self):
    return len(self.x)


class FashionMNIST:
  """FashionMNIST dataset."""

  def __init__(self, train=True):
    mnist = torchvision.datasets.FashionMNIST('/tmp/fashionmnist', download=True, train=train)
    self.x = mnist.data
    self.y = mnist.targets

  def __getitem__(self, idx):
    return self.x[idx] / 255., self.y[idx]

  def __len__(self):
    return len(self.x)


class CIFAR10:
  """CIFAR10 dataset."""

  def __init__(self, train=True):
    cifar10 = torchvision.datasets.CIFAR10('/tmp/cifar10', download=True, train=train)
    self.x = cifar10.data
    self.y = cifar10.targets

  def __getitem__(self, idx):
    return self.x[idx] / 255., self.y[idx]

  def __len__(self):
    return len(self.x)


class CIFAR10C:
  """CIFAR10-C dataset (Hendrycks et al., 2019)."""

  corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
  ]

  url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
  download_dir = '/tmp/cifar10-c'

  def __init__(self, corruption=None):
    if os.path.exists(self.download_dir):
      print('CIFAR10-C already downloaded')
    else:
      download_and_extract_archive(self.url, self.download_dir)

    # Load data
    path_fn = lambda c: os.path.join(self.download_dir, 'CIFAR-10-C', f'{c}.npy')
    if corruption is None:
      paths = [path_fn(c) for c in self.corruptions]
    elif isinstance(corruption, str):
      assert corruption in self.corruptions, f'corruption must be in {self.corruptions}'
      paths = [path_fn(corruption)]
    elif isinstance(corruption, list):
      assert all(c in self.corruptions for c in corruption), f'corruptions must be in {self.corruptions}'
      paths = [path_fn(c) for c in corruption]
    else:
      raise ValueError('corruption must be None, str, or list')

    label_path = path_fn('labels')
    labels = np.load(label_path)
    self.data = []
    self.targets = []
    for path in paths:
      assert os.path.exists(path), f'{path} does not exist'
      data = np.load(path)
      self.data.append(data)
      self.targets.append(labels)

    self.data = np.concatenate(self.data, axis=0)
    self.targets = np.concatenate(self.targets, axis=0)
    assert len(self.data) == len(self.targets)

  def __getitem__(self, idx):
    return self.data[idx] / 255., self.targets[idx]

  def __len__(self):
    return len(self.data)
