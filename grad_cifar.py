import itertools

import matplotlib.pyplot as plt
import torch

import gc
import os
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from robustbench.utils import load_model
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy
from torchmetrics import Accuracy
from cifar10_models.resnet import resnet18
from torchvision import models
from mlp_mixer_pytorch import MLPMixer

import dataset
import model
import utils
import train

import argparse

parser = argparse.ArgumentParser()

# Add arguments to the parser
parser.add_argument("--n_step", type=int, default=500, help="number of gradient acc steps")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--lr", type=int, default=1e-4, help="learning rate")
parser.add_argument("--data_t0", type=str, default="cifar10", help="task 0 dataset")
parser.add_argument("--data_t1", type=str, default="cifar10c", help="task 1 dataset")
parser.add_argument("--model", type=str, help="model (resnet, mixer, cifarresnet)")
parser.add_argument("--pretrained", type=bool, default=False, help="Pretrained resnet or not, for mixer this should always be false")
parser.add_argument("--cifarcsmall", type=bool, default=False, help="use small cifarc")
parser.add_argument("--mixer_dim", type=int, default=512, help="mixer dim")
parser.add_argument("--mixer_depth", type=int, default=12, help="mixer depth")

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
n_step = args.n_step
batch_size = args.batch_size
lr = args.lr
data_t0 = args.data_t0
data_t1 = args.data_t1
model = args.model
pretrained = args.pretrained
cifarcsmall = args.cifarcsmall
mixer_dim = args.mixer_dim
mixer_depth = args.mixer_depth
# Set device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Set MISC
accuracy = Accuracy().to(DEVICE)
corruptions=["gaussian_noise"]
t0_permute = False
t1_permute = False
# Get Data
if data_t0 == "cifar10":
    t0_permute = True
    t0_train = dataset.CIFAR10()
    t0_valid = dataset.CIFAR10(train=False)
if data_t0 == "cifar10c":
    if not cifarcsmall:
        t0_permute = True
    t0_train = dataset.CIFAR10C(corruptions, flipped=False, small=cifarcsmall, ratio=0.2)
    t0_valid = cifarc_valid = dataset.CIFAR10C(corruptions, flipped=False, train=False, small=cifarcsmall, ratio=0.2)
if data_t0 == "cifar10cf":
    if not cifarcsmall:
        t0_permute = True
    t0_train = dataset.CIFAR10C(corruptions, flipped=True, small=cifarcsmall, ratio=0.2)
    t0_valid = cifarc_valid = dataset.CIFAR10C(corruptions, flipped=True, train=False, small=cifarcsmall, ratio=0.2)

if data_t1 == "cifar10":
    t1_permute = True
    t1_train = dataset.CIFAR10()
    t1_valid = dataset.CIFAR10(train=False)
if data_t1 == "cifar10c":
    if not cifarcsmall:
        t1_permute = True
    t1_train = dataset.CIFAR10C(corruptions, flipped=False, small=cifarcsmall, ratio=0.2)
    t1_valid = cifarc_valid = dataset.CIFAR10C(corruptions, flipped=False, train=False, small=cifarcsmall, ratio=0.2)
if data_t1 == "cifar10cf":
    if not cifarcsmall:
        t1_permute = True
    t1_train = dataset.CIFAR10C(corruptions, flipped=True, small=cifarcsmall, ratio=0.2)
    t1_valid = cifarc_valid = dataset.CIFAR10C(corruptions, flipped=True, train=False, small=cifarcsmall, ratio=0.2)

t0_train_loader = DataLoader(t0_train, batch_size=batch_size, shuffle=True, drop_last=True)
t0_valid_loader = DataLoader(t0_valid, batch_size=batch_size, shuffle=True, drop_last=True)
t0_iter = itertools.cycle(t0_train_loader)

t1_train_loader = DataLoader(t1_train, batch_size=batch_size, shuffle=True, drop_last=True)
t1_valid_loader = DataLoader(t1_valid, batch_size=batch_size, shuffle=True, drop_last=True)
t1_iter = itertools.cycle(t1_train_loader)

# Set Tasks
tasks = {
    't0': {
        'train_iter': t0_iter,
        'eval_ds': t0_valid_loader,
        'loss': lambda logits, labels: F.cross_entropy(logits, labels),
        'predict': lambda logits: logits,
        'metric': lambda yh, y: accuracy(yh, y),
    },
    't1': {
        'train_iter': t1_iter,
        'eval_ds': t1_valid_loader,
        'loss': lambda logits, labels: F.cross_entropy(logits, labels),
        'predict': lambda logits: logits,
        'metric': lambda yh, y: accuracy(yh, y),
    },
}

# Set model
if model == "resnet18":
    backbone = resnet18(pretrained=pretrained)
    # backbone = nn.Sequential(*list(backbone.children())[:-1])
if model == "mixer":
    backbone = MLPMixer(
                    image_size = 32,
                    channels = 3,
                    patch_size = 8,
                    dim = mixer_dim,
                    depth = mixer_depth,
                    num_classes = 10
                )
    # backbone = nn.Sequential(*list(backbone.children())[:-1])
if model == "cifarcresnet18":
    backbone = load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf') 
if model == "imagenetresnet18":
    backbone = models.resnet18(pretrained=True)

if model != "cifarcresnet18":
    backbone = nn.Sequential(*list(backbone.children())[:-1])


class Shared_MTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = backbone
        # print(self.net)
        if model != "cifarcresnet18":
            self.t0_head = nn.Linear(512, 10)
            self.t1_head = nn.Linear(512, 10)
        else:
            self.t0_head = nn.Linear(10, 10)
            self.t1_head = nn.Linear(10, 10)
    
    def forward(self, x, task):

        if task == "t0":
            if t0_permute:
                x = torch.permute(x, (0, 3, 1, 2)).float()
            # if model == "resnet18":
            if "resnet18" in model and "cifarc" not in model:
                bb = self.net(x)
                # print(bb.shape)
                b, h, _, _ = bb.shape
                logits = self.t0_head(bb.view(b, h))
            else:
                logits = self.t0_head(self.net(x))
        if task == "t1":
            if t1_permute:
                x = torch.permute(x, (0, 3, 1, 2)).float()
            # if model == "resnet18":
            if "resnet18" in model and "cifarc" not in model:
                bb = self.net(x)
                b, h, _, _ = bb.shape
                logits = self.t1_head(bb.view(b, h))
            else:
                logits = self.t1_head(self.net(x))
        return logits

shared_net = Shared_MTL()
shared_net.to(DEVICE)

param_keys = []

for k, v in shared_net.named_parameters():
    # print(k)
    if "head" in k:
        break
    # if "conv2" in k:
    param_keys.append(k)

# s = ""
# for p in param_keys:
#     s += p + ","
# print(s)

grads = train.get_gradients(
    model=shared_net,
    tasks=tasks, 
    steps=n_step, 
    lr=lr,
    DEVICE=DEVICE,
    param_keys=param_keys
)

heuristic_results = {}

# # plots
n_rows = math.ceil(len(param_keys)/3)
n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
path = model + "_" + data_t0 + "_" + data_t1 + "_pretrained_" + str(pretrained) + "_small_" + str(cifarcsmall)
if model == "mixer":
    path = path + "_" + str(mixer_depth)
if not os.path.isdir(path):
    os.mkdir(path)
for i, key in enumerate(param_keys):
    # heuristics computations
    cosine = torch.sum(F.normalize(utils.stack_grad(grads, 't1', key), dim=-1) * F.normalize(utils.stack_grad(grads, 't0', key), dim=-1), dim=-1)
    smooth_cos = utils.low_pass_filter(cosine[None], filter_size=10)[0][0]
    
    
    torch.save(cosine, path + "/" + "cosine_" + key + ".pt")
    torch.save(smooth_cos, path + "/" + "smooth_cos_" + key + ".pt")

    avg_smoothed_cos = smooth_cos.mean()
    std_cos = cosine.std()
    
    heuristic_results[key] = {'avg_cos': avg_smoothed_cos, 'std_cos': std_cos}
      
    # plot
    row, col = i // n_cols, i % n_cols
    ax = axes[row][col]    
    ax.set_title(key)
    if col == 0:
        ax.set_ylabel('Gradient cosine similarity')
    if row == n_rows - 1:
        ax.set_xlabel('Step')
    ax.set_ylim([-1.1, 1.1])
    ax.plot(cosine, color='teal', alpha=0.2)
    ax.plot(smooth_cos, color='teal')

fig.savefig(path + "/fig.png")