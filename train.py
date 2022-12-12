import copy

import torch
from torch import nn
from tqdm import tqdm

import utils


def get_separated_gradients(*, model, tasks, steps=200, lr=3e-4):
  """Compute per-task per-model gradients."""

  models = {
      task_name: copy.deepcopy(model)
      for task_name in tasks
  }
  opts = {
      task_name: torch.optim.SGD(models[task_name].parameters(), lr=lr)
      for task_name in models
  }

  step = 0
  losses = []
  grads = []
  pbar = tqdm(total=steps)
  while step < steps:
    task_losses = {}
    task_grads = {}

    for task_name in tasks.keys():
      opt = opts[task_name]
      model = models[task_name]

      opt.zero_grad()
      task_iter = tasks[task_name]['train_iter']
      task_loss_fn = tasks[task_name]['loss']

      batch = next(task_iter)
      x, y = batch
      pred = model(x, task_name)
      loss = task_loss_fn(pred, y)

      running_grads = utils.clone_grads(model)
      loss.backward()
      task_grads[task_name] = utils.sub_state_dicts(utils.clone_grads(model), running_grads)
      task_losses[task_name] = loss.item()

      opt.step()

    losses.append(task_losses)
    grads.append(task_grads)

    step += 1
    pbar.update(1)
  pbar.close()

  return grads, losses


def get_gradients_sam(*, model, tasks, steps=200, lr=3e-4):
  from thirdparty.sam_pytorch import sam

  opt = sam.SAMSGD(model.parameters(), lr=lr, rho=0.05)
  # torch.optim.Adam(model.parameters(), lr=lr)

  step = 0
  losses = []
  grads = []
  pbar = tqdm(total=steps)
  while step < steps:
    task_losses = {}
    task_grads = {}

    def closure():
      opt.zero_grad()
      total_loss = 0
      for task_name in tasks.keys():
        task_iter = tasks[task_name]['train_iter']
        task_loss_fn = tasks[task_name]['loss']

        batch = next(task_iter)
        x, y = batch
        pred = model(x, task_name)
        loss = task_loss_fn(pred, y)
        total_loss += loss

        # avoid gradient accumulation bugs
        with torch.no_grad():
          running_grads = utils.clone_grads(model)
        loss.backward()
        with torch.no_grad():
          task_grads[task_name] = utils.sub_state_dicts(utils.clone_grads(model), running_grads)
          task_losses[task_name] = loss.item()

      return total_loss

    loss = opt.step(closure)
    # task_losses = {task_name: loss.item() for task_name in tasks.keys()}

    losses.append(task_losses)
    grads.append(task_grads)

    step += 1
    pbar.update(1)
  pbar.close()

  return grads, losses, []



def get_gradients(*, model, tasks, steps=200, lr=3e-4):
  """Compute per-task gradients."""

  opt = torch.optim.Adam(model.parameters(), lr=lr)

  step = 0
  losses = []
  grads = []
  rgn = []
  pbar = tqdm(total=steps)
  while step < steps:
    task_losses = {}
    task_grads = {}
    task_rgn = {}

    opt.zero_grad()
    for task_name in tasks.keys():
      task_iter = tasks[task_name]['train_iter']
      task_loss_fn = tasks[task_name]['loss']

      batch = next(task_iter)
      x, y = batch
      pred = model(x, task_name)
      loss = task_loss_fn(pred, y)

      # avoid gradient accumulation bugs
      with torch.no_grad():
        running_grads = utils.clone_grads(model)
      loss.backward()
      with torch.no_grad():
        task_grads[task_name] = utils.sub_state_dicts(utils.clone_grads(model), running_grads)

        # compute rgn
        task_rgn[task_name] = {}
        for pn, p in model.named_parameters():
          gi = task_grads[task_name][pn]
          if isinstance(gi, int) and gi == 0:
            task_rgn[task_name][pn] = 0
          else:
            task_rgn[task_name][pn] = torch.norm(gi) / torch.norm(p)

        task_losses[task_name] = loss.item()

    opt.step()

    losses.append(task_losses)
    grads.append(task_grads)
    rgn.append(task_rgn)

    step += 1
    pbar.update(1)
  pbar.close()

  return grads, losses, rgn


def train_and_evaluate(*, model, tasks, steps=1000, lr=3e-4, eval_every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

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
      task_eval_losses = {}
      task_eval_metrics = {}

      opt.zero_grad()
      for task_name in tasks.keys():
        task = tasks[task_name]
        task_iter = task['train_iter']
        loss_fn = task['loss']
        predict_fn = task['predict']
        metric_fn = task['metric']

        batch = next(task_iter)
        x, y = batch
        logits = model(x, task_name)
        loss = loss_fn(logits, y)
        loss.backward()
        task_losses[task_name] = loss.item()
        task_metrics[task_name] = metric_fn(predict_fn(logits), y)

      opt.step()

      losses.append((step, task_losses))
      metrics.append((step, task_metrics))

      if step % eval_every == 0:
          model.eval()
          for task_name in tasks.keys():
            task = tasks[task_name]
            eval_ds = task['eval_ds']
            loss_fn = task['loss']
            predict_fn = task['predict']
            metric_fn = task['metric']

            x = []
            y = []
            for idx in range(len(eval_ds)):
              x_i, y_i = eval_ds[idx]
              x.append(x_i)
              y.append(y_i)
            x = torch.stack(x)
            y = torch.stack(y)

            logits = model(x, task_name)
            task_eval_losses[task_name] = loss_fn(logits, y).item()
            task_eval_metrics[task_name] = metric_fn(predict_fn(logits), y)

          eval_losses.append((step, task_eval_losses))
          eval_metrics.append((step, task_eval_metrics))

      step += 1
      pbar.update(1)
    pbar.close()
    return losses, metrics, eval_losses, eval_metrics


def train_and_evaluate_sam(*, model, tasks, steps=1000, lr=3e-4, eval_every=100):
    from thirdparty.sam_pytorch import sam

    opt = sam.SAMSGD(model.parameters(), lr=lr, rho=0.05)

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
      task_eval_losses = {}
      task_eval_metrics = {}

      def closure():
        opt.zero_grad()
        total_loss = 0
        for task_name in tasks.keys():
          task = tasks[task_name]
          task_iter = task['train_iter']
          loss_fn = task['loss']
          predict_fn = task['predict']
          metric_fn = task['metric']

          batch = next(task_iter)
          x, y = batch
          logits = model(x, task_name)
          loss = loss_fn(logits, y)
          total_loss += loss
          loss.backward()
          task_losses[task_name] = loss.item()
          task_metrics[task_name] = metric_fn(predict_fn(logits), y)

        return loss

      opt.step(closure)

      losses.append((step, task_losses))
      metrics.append((step, task_metrics))

      if step % eval_every == 0:
          model.eval()
          for task_name in tasks.keys():
            task = tasks[task_name]
            eval_ds = task['eval_ds']
            loss_fn = task['loss']
            predict_fn = task['predict']
            metric_fn = task['metric']

            x = []
            y = []
            for idx in range(len(eval_ds)):
              x_i, y_i = eval_ds[idx]
              x.append(x_i)
              y.append(y_i)
            x = torch.stack(x)
            y = torch.stack(y)

            logits = model(x, task_name)
            task_eval_losses[task_name] = loss_fn(logits, y).item()
            task_eval_metrics[task_name] = metric_fn(predict_fn(logits), y)

          eval_losses.append((step, task_eval_losses))
          eval_metrics.append((step, task_eval_metrics))

      step += 1
      pbar.update(1)
    pbar.close()
    return losses, metrics, eval_losses, eval_metrics
