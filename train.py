import torch
from torch import nn
from tqdm import tqdm

import utils


def get_gradients(*, model, tasks, steps=200, lr=3e-4, DEVICE=None, param_keys=[]):
  """Compute per-task gradients."""

  opt = torch.optim.Adam(model.parameters(), lr=lr)

  if DEVICE is None:
      DEVICE = model.device

  step = 0
  losses = []
  grads = []
  pbar = tqdm(total=steps)
  while step < steps:
    task_losses = {}
    task_grads = {}

    opt.zero_grad()
    for task_name in tasks.keys():
      task_iter = tasks[task_name]['train_iter']
      task_loss_fn = tasks[task_name]['loss']

      batch = next(task_iter)
      x, y = batch
      x, y = x.to(DEVICE), y.to(DEVICE)
      pred = model(x, task_name)
      loss = task_loss_fn(pred, y)

      # avoid gradient accumulation bugs
      running_grads = utils.clone_grads(model, param_keys)
      loss.backward()
      task_grads[task_name] = utils.sub_state_dicts(utils.clone_grads(model, param_keys), running_grads)
      task_losses[task_name] = loss.item()

    taskLossesValuesText = [f"{x:.3f}" for x in task_losses.values()]
    pbar.set_description(f"Losses: {'.'.join(taskLossesValuesText)}")

    opt.step()

    losses.append(task_losses)
    grads.append(task_grads)

    step += 1
    pbar.update(1)
  pbar.close()

  return grads


def train_and_evaluate(*, model, tasks, steps=1000, lr=3e-4, eval_every=100, DEVICE=None):
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
      task_eval_losses = {"t0": 0, "t1": 0}
      task_eval_metrics = {"t0": 0, "t1": 0}

      opt.zero_grad()
      for task_name in tasks.keys():
        task = tasks[task_name]
        task_iter = task['train_iter']
        loss_fn = task['loss']
        predict_fn = task['predict']
        metric_fn = task['metric']

        batch = next(task_iter)
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x, task_name)
        loss = loss_fn(logits, y)
        loss.backward()
        task_losses[task_name] = loss.item()
        task_metrics[task_name] = metric_fn(predict_fn(logits), y).clone().detach().cpu()

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
            for batch in task["eval_ds"]:
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
              task_eval_metrics[task_name] += metric_fn(predict_fn(model(x, task_name)), y).clone().detach().cpu()
              # print(metric_fn(predict_fn(model(x, task_name)), y).clone().detach().cpu())
            task_eval_losses[task_name] = task_eval_losses[task_name]/len(task["eval_ds"])
            task_eval_metrics[task_name] = task_eval_metrics[task_name]/len(task["eval_ds"])
          eval_losses.append((step, task_eval_losses))
          eval_metrics.append((step, task_eval_metrics))

      step += 1
      pbar.update(1)
    pbar.close()
    return losses, metrics, eval_losses, eval_metrics
