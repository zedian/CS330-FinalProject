from torch import nn
import torch


def evaluate(*, model, val_loader, metrics=None):
  pass


def train_one_epoch(*, model, optimizer, task_loaders, metrics=None):
  assert isinstance(task_loaders, list)
  pass


def get_gradients(*. model, optimizer, task_loaders):
  mtl.train()
  train_loss = []
  for epoch in range(EPOCHS):
      pbar = tqdm(train_iter)
      for batch in pbar:
          im, t0_labels, t1_labels = batch
          loss = mtl.loss(im, t0_labels.float(), t1_labels.float())

          opt.zero_grad()
          names, params = zip(*mtl.named_parameters())
          grads = torch.autograd.grad(loss['loss/t0'], params, allow_unused=True, retain_graph=True)
          t0_grads = list(zip(names, grads))
          gradients['t0'].append(dict(t0_grads))

          opt.zero_grad()
          names, params = zip(*mtl.named_parameters())
          grads = torch.autograd.grad(loss['loss/t1'], params, allow_unused=True, retain_graph=True)
          t1_grads = list(zip(names, grads))
          gradients['t1'].append(dict(t1_grads))

          # I don't think this is double accumulating?
          opt.zero_grad()
          loss['loss/total'].backward()
          train_loss.append(loss['loss/total'].item())
          pbar.set_description('Loss %f' % loss['loss/total'].item())
          opt.step()


def get_parameter_sharing_spec(*, model):
  pass



def train_mtl(
    *,
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    epochs=5,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_metrics = {'loss/total': [],
                     'loss/t0': [],
                     'loss/t1': [],
                     'acc/t0': [],
                     'acc/t1': [],
                    }
    val_metrics = {'loss/total': [],
                     'loss/t0': [],
                     'loss/t1': [],
                     'acc/t0': [],
                     'acc/t1': [],
                    }

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, total=len(train_loader)):
            im, t0_labels, t1_labels = batch

            loss = model.loss(im, t0_labels.float(), t1_labels.float())
            total_loss = loss['loss/total']

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # loss metrics
            for k in train_metrics.keys():
                if k in loss:
                    train_metrics[k].append(loss[k].item())

            # accuracy metrics
            t0_logits, t1_logits = model.forward(im)
            t0_pred = nn.functional.sigmoid(t0_logits) > 0.5
            t1_pred = nn.functional.sigmoid(t1_logits) > 0.5
            train_metrics['acc/t0'].append(accuracy(t0_pred[:, 0], t0_labels.bool()))
            train_metrics['acc/t1'].append(accuracy(t1_pred[:, 0], t1_labels.bool()))


        t0_val_loss = 0
        t1_val_loss = 0
        t0_val_acc = 0
        t1_val_acc = 0
        val_count = 0

        model.eval()
        for batch in val_loader:
            im, t0_labels, t1_labels = batch
            val_loss = model.loss(im, t0_labels.float(), t1_labels.float())

            val_count += len(im)
            t0_val_loss += val_loss['loss/t0'].item() * len(im)
            t1_val_loss += val_loss['loss/t1'].item() * len(im)

            # accuracy metrics
            t0_logits, t1_logits = model.forward(im)
            t0_pred = torch.sigmoid(t0_logits) > 0.5
            t1_pred = torch.sigmoid(t1_logits) > 0.5
            t0_val_acc += accuracy(t0_pred[:, 0], t0_labels.bool()).item() * len(im)
            t1_val_acc += accuracy(t1_pred[:, 0], t1_labels.bool()).item() * len(im)

        val_metrics['loss/t0'].append(t0_val_loss / val_count)
        val_metrics['loss/t1'].append(t1_val_loss / val_count)
        val_metrics['acc/t0'].append(t0_val_acc / val_count)
        val_metrics['acc/t1'].append(t1_val_acc / val_count)

    return train_metrics, val_metrics