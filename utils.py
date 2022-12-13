import torch
import torch.nn.functional as F


def accuracy(y_hat, y):
    assert y_hat.shape == y.shape, (y_hat.shape, y.shape)
    n_classes = y_hat.shape[-1]
    y_hat = y_hat.view(-1, n_classes)
    y = y.view(-1, n_classes)
    n_correct = torch.sum(y_hat == y)
    return n_correct / y_hat.numel()


def grad_dict(loss, model):
    names, params = zip(*model.named_parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    zipped_grads = list(zip(names, grads))
    return dict(zipped_grads)


def clone_grads(model):
    names, params = zip(*model.named_parameters())
    grads = [p.grad if p.grad is None else p.grad.clone() for p in params]
    return dict(zip(names, grads))


def sub_state_dicts(a, b):
    assert a.keys() == b.keys()
    a_vals = a.values()
    b_vals = b.values()
    return {k: (v1 if v1 is not None else 0) - (v2 if v2 is not None else 0)
            for k, v1, v2 in zip(a.keys(), a_vals, b_vals)}


def stack_grad(grad_dict_list, task_name, param_name, flatten=True):
    grads = [g[task_name][param_name] for g in grad_dict_list]
    return torch.stack(grads).view(len(grads), -1)


def low_pass_filter(x, filter_size=25):
    x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    x_smooth = F.conv1d(x[None], torch.ones(1, 1, filter_size) / filter_size)
    return x_smooth.numpy()


################################################
# Synthetic Data Generation
# Source: https://arxiv.org/pdf/1910.04915v2.pdf
################################################

import math
import numpy as np
import scipy.stats


def generate_label(input_features, task_weights, alphas, betas, noise_std):
  """Generate a single label.

  The label is generated according to the definition in "Modeling Task
  Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts".

  Args:
    input_features: (numpy array of shape [`dim`]) input features.
    task_weights: (numpy array of shape [`dim`]) weights that define the task.
    alphas: (sequence of `m` floats) multipliers for the non-linear part
      of the label.
    betas: (sequence of `m` floats) biases for the non-linear part of the label.
    noise_std: (float) standard deviation for the noise added to the label.

  Returns:
    Generated label as a numpy scalar.
  """
  dot = np.dot(task_weights, input_features)
  non_linear_parts = [
      math.sin(alpha * dot + beta) for alpha, beta in zip(alphas, betas)]
  noise = np.random.normal(scale=noise_std)

  return dot + sum(non_linear_parts) + noise


def generate_task_pairs(
    num_task_pairs,
    num_samples,
    dim,
    relatedness,
    c=1.0,
    m=10,
    noise_std=0.01,
    shuffle_task_data=False):
  """Generate data for `num_task_pairs` pairs of tasks.

  Relatedness inside each pair of tasks can be configured, while the relatedness
  of tasks from different pairs will be low.

  Args:
    num_task_pairs: (int) number of pairs of tasks.
    num_samples: (int) number of samples to generate for each of the tasks.
    dim: (int) dimension for the synthetically generated data.
    relatedness: (float) between 0 and 1; the higher the more related the
      pairs of tasks will be.
    c: (float) scale for the linear part of the label.
    m: (int) number of summands in the non-linear part of the label.
    noise_std: (float) standard deviation for the noise added to the label.
    shuffle_task_data: (bool) if set to True, the features and labels for each
      task are randomly and independently reshuffled.

  Returns:
    Features and labels for all pairs of tasks as numpy arrays.

    If `shuffle_task_data` is set to False, then the returned feature
    array is the same for all tasks, so it has shape [`num_samples`, `dim`].
    The shape of the labels array is [2 * `num_task_pairs`, `num_samples`],
    where the first two tasks constitute the first pair, the next two
    the second pair, and so on.

    If `shuffle_task_data` is set to True, then a list of length
    2 * `num_task_pairs` is returned. Each element of the list corresponds to
    a single task, and contains features of shape [`num_samples`, `dim`], and
    labels of shape [`num_samples`].
  """
  assert 0.0 <= relatedness <= 1.0

  num_tasks = 2 * num_task_pairs

  # We need to generate `num_tasks` orthogonal basis vectors
  assert dim >= num_tasks

  # The original paper does not provide the values for these sequences
  alphas = range(1, m + 1)
  betas = [i ** 2 for i in range(m)]

  # Generate a random [`dim`, `dim`] orthogonal matrix, and select `num_tasks`
  # of first rows
  basis_vectors = scipy.stats.ortho_group.rvs(dim)[:num_tasks]
  basis_vectors = np.reshape(basis_vectors, (num_task_pairs, 2, dim))

  xs = np.random.normal(size=(num_samples, dim))
  task_ys = []
  task_w1s = []
  task_w2s = []

  for u1, u2 in basis_vectors:
    w1 = c * u1
    w2 = c * (relatedness * u1 + math.sqrt(1 - relatedness ** 2) * u2)

    for w in [w1, w2]:
      labels = [generate_label(x, w, alphas, betas, noise_std) for x in xs]
      labels = [np.expand_dims(label, axis=0) for label in labels]

      task_ys.append(np.concatenate(labels))

    task_w1s.append(w1)
    task_w2s.append(w2)

  if shuffle_task_data:
    data = []

    for y in task_ys:
      perm = np.random.permutation(len(xs))
      data.append((xs[perm], y[perm]))

    return data
  else:
    return xs, np.stack(task_ys)
