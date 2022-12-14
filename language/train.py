import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import numpy as N

import utils


def get_gradients(*,
                  model, tasks, steps=None, lr=3e-4, DEVICE=None, param_keys=[]):
  """Compute per-task gradients."""

  opt = torch.optim.Adam(model.parameters(), lr=lr)

  if DEVICE is None:
      DEVICE = model.device

  if not steps:
    steps = max(len(task['train_gen']) for task in tasks.values())
  losses = []
  cosines = []
  pbar = tqdm(total=steps)
  taskName1, taskName2 = list(tasks.keys())
  filteredParamKeys = []

  for step in trange(steps):
    task_losses = {}
    task_grads = {}

    opt.zero_grad()
    for task_name,task in tasks.items():
      task_loss_fn = task['loss']

      batch = task['train_gen'].get_next_batch()
      x, y = batch
      x, y = x.to(DEVICE), y.to(DEVICE)
      pred = model(x, task_name)
      loss = task_loss_fn(pred, y)

      # avoid gradient accumulation bugs
      running_grads = utils.clone_grads(model, param_keys)
      loss.backward()
      task_grads[task_name] = utils.sub_state_dicts(utils.clone_grads(model, param_keys), running_grads)
      task_losses[task_name] = loss.item()

    textLoss = ", ".join(f"{x:.3f}" for x in task_losses.values())
    pbar.set_description(f"Losses: {textLoss}")

    opt.step()

    cosine = []
    filteredParamKeys = []
    for k in param_keys:
        g1 = task_grads[taskName1][k]
        g2 = task_grads[taskName2][k]
        if isinstance(g1, int) or isinstance(g2, int):
            # Represents a task head. discard.
            continue

        c = torch.sum(F.normalize(torch.flatten(g1), dim=-1) *
                      F.normalize(torch.flatten(g2), dim=-1)).item()
        cosine.append(c)
        filteredParamKeys.append(k)

    losses.append(task_losses)
    cosines.append(cosine)

  pbar.close()

  return N.array(cosines), filteredParamKeys


def train_and_evaluate(*,
                       model, tasks,
                       steps=1000, lr=3e-4, eval_every=100,
                       DEVICE=None, writers=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if DEVICE is None:
        DEVICE = model.device

    step = 0

    losses = []
    metrics = []
    eval_losses = []
    eval_metrics = []
    pbar = tqdm(total=steps)
    while step < steps:
      model.train()
      task_losses = {}
      task_metrics = {}
      task_eval_losses = {key:0 for key in tasks.keys()}
      task_eval_metrics = {key:0 for key in tasks.keys()}

      opt.zero_grad()
      for task_name in tasks.keys():
        task = tasks[task_name]
        loss_fn = task['loss']
        predict_fn = task['predict']
        metric_fn = task['metric']

        batch = task['train_gen'].get_next_batch()
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x, task_name)
        loss = loss_fn(logits, y)
        loss.backward()
        task_losses[task_name] = loss.item()
        task_metrics[task_name] = metric_fn(logits, y).clone().detach().cpu()

      opt.step()

      losses.append((step, task_losses))
      metrics.append((step, task_metrics))

      if step % eval_every == 0:
          model.eval()
          for task_name in tasks.keys():
            task = tasks[task_name]
            loss_fn = task['loss']
            predict_fn = task['predict']
            metric_fn = task['metric']
            for iBatch in range(len(task["val_gen"])):
              batch = task["val_gen"][iBatch]
              x, y = batch
              x, y = x.to(DEVICE), y.to(DEVICE)
              # x = []
              # y = []
              # for idx in range(len(eval_ds)):
              #   x_i, y_i = eval_ds[idx]
              #   x.append(x_i)
              #   y.append(y_i)
              # x = torch.stack(x)
              # y = torch.stack(y)
              # x = torch.Tensor(x).to(DEVICE)
              # y = torch.Tensor(y).type(torch.LongTensor).to(DEVICE)
              # print(y)
              # print(model(x, task_name))
              task_eval_losses[task_name] += loss_fn(model(x, task_name), y).item()
              task_eval_metrics[task_name] += metric_fn(model(x, task_name), y).clone().detach().cpu()
            task_eval_losses[task_name] = task_eval_losses[task_name]/len(task["val_gen"])
            task_eval_metrics[task_name] = task_eval_metrics[task_name]/len(task["val_gen"])
          eval_losses.append((step, task_eval_losses))
          eval_metrics.append((step, task_eval_metrics))
          if writers:
            for taskKey in tasks.keys():
              writers[taskKey].add_scalar(f"Loss/eval", task_eval_losses[taskKey], step)
              writers[taskKey].add_scalar(f"Metric/eval", task_eval_metrics[taskKey], step)

      step += 1
      textLoss = ", ".join(f"{x:.3f}" for x in task_losses.values())
      textMetric = ", ".join(f"{x:.3f}" for x in task_metrics.values())
      pbar.set_description(f"Losses: {textLoss}; Metrics: {textMetric}")
      pbar.update(1)

      if writers:
        for taskKey in tasks.keys():
          writers[taskKey].add_scalar(f"Loss/train", task_losses[taskKey], step)
          writers[taskKey].add_scalar(f"Metric/train", task_metrics[taskKey], step)
    pbar.close()
    return losses, metrics, eval_losses, eval_metrics
