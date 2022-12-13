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


def clone_grads(model, param_keys):
    names, params = zip(*model.named_parameters())
    key_params = [param for n, param in zip(names, params) if n in param_keys]
    grads = [p.grad if p.grad is None else p.grad.clone().detach().cpu() for p in key_params]
    return dict(zip(param_keys, grads))


def sub_state_dicts(a, b):
    assert a.keys() == b.keys()
    a_vals = a.values()
    b_vals = b.values()
    return {k: (v1 if v1 is not None else 0) - (v2 if v2 is not None else 0)
            for k, v1, v2 in zip(a.keys(), a_vals, b_vals)}


def stack_grad(grad_dict_list, task_name, param_name, flatten=True):
    grads = [g[task_name][param_name] for g in grad_dict_list]
    return torch.stack(grads).view(len(grads), -1)
    # return np.array(grads)


def low_pass_filter(x, filter_size=25):
    x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    x_smooth = F.conv1d(x[None], torch.ones(1, 1, filter_size) / filter_size)
    return x_smooth.numpy()
