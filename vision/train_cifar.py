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
parser.add_argument("--positive", type=bool, default=False, help="Strictly positive grad")

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
n_step = args.n_step
batch_size = args.batch_size
lr = args.lr
data_t0 = args.data_t0
data_t1 = args.data_t1
used_model = args.model
pretrained = args.pretrained
cifarcsmall = args.cifarcsmall
mixer_dim = args.mixer_dim
mixer_depth = args.mixer_depth
positive = args.positive
print(positive)

path = used_model + "_" + data_t0 + "_" + data_t1 + "_pretrained_" + str(pretrained) + "_small_" + str(cifarcsmall)
if used_model == "mixer":
    path += "_" + str(mixer_depth)

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
if used_model == "resnet18":
    backbone_shared = resnet18(pretrained=pretrained)
    # backbone2_shared = resnet18(pretrained=pretrained)
    backbone_separate = resnet18(pretrained=pretrained)
    backbone2_separate = resnet18(pretrained=pretrained)
    backbone_surgical = resnet18(pretrained=pretrained)
    # backbone2_surgical = resnet18(pretrained=pretrained)
    # backbone = nn.Sequential(*list(backbone.children())[:-1])
if used_model == "mixer":
    backbone_shared = MLPMixer(
                    image_size = 32,
                    channels = 3,
                    patch_size = 8,
                    dim = mixer_dim,
                    depth = mixer_depth,
                    num_classes = 10
                )
    # backbone2_shared = MLPMixer(
    #                 image_size = 32,
    #                 channels = 3,
    #                 patch_size = 8,
    #                 dim = mixer_dim,
    #                 depth = mixer_depth,
    #                 num_classes = 10
    #             )
    backbone_separate = MLPMixer(
                    image_size = 32,
                    channels = 3,
                    patch_size = 8,
                    dim = mixer_dim,
                    depth = mixer_depth,
                    num_classes = 10
                )
    backbone2_separate = MLPMixer(
                    image_size = 32,
                    channels = 3,
                    patch_size = 8,
                    dim = mixer_dim,
                    depth = mixer_depth,
                    num_classes = 10
                )
    backbone_surgical = MLPMixer(
                    image_size = 32,
                    channels = 3,
                    patch_size = 8,
                    dim = mixer_dim,
                    depth = mixer_depth,
                    num_classes = 10
                )
    # backbone2_surgical = MLPMixer(
    #                 image_size = 32,
    #                 channels = 3,
    #                 patch_size = 8,
    #                 dim = mixer_dim,
    #                 depth = mixer_depth,
    #                 num_classes = 10
    #             )
    # backbone = nn.Sequential(*list(backbone.children())[:-1])
if used_model == "imagenetresnet18":
    backbone = models.resnet18(pretrained=True)
    backbone2 = models.resnet18(pretrained=True)

if used_model == "cifarcresnet18":
    backbone_shared = load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf') 
    backbone_separate = load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf') 
    backbone2_separate = load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf') 
    backbone_surgical = load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf') 
else:
    backbone_shared = nn.Sequential(*list(backbone_shared.children())[:-1])
    # backbone2_shared = nn.Sequential(*list(backbone2_shared.children())[:-1])
    backbone_separate = nn.Sequential(*list(backbone_separate.children())[:-1])
    backbone2_separate = nn.Sequential(*list(backbone2_separate.children())[:-1])
    backbone_surgical = nn.Sequential(*list(backbone_surgical.children())[:-1])
    # backbone2_surgical = nn.Sequential(*list(backbone2_surgical.children())[:-1])

class Shared_MTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = backbone_shared
        # print(self.net)
        if used_model == "cifarcresnet18":
            self.t0_head = nn.Linear(10, 10)
            self.t1_head = nn.Linear(10, 10)
        else:
            self.t0_head = nn.Linear(512, 10)
            self.t1_head = nn.Linear(512, 10)
    
    def forward(self, x, task):

        if task == "t0":
            if t0_permute:
                x = torch.permute(x, (0, 3, 1, 2)).float()
            # if model == "resnet18":
            if "resnet18" in used_model and "cifar" not in used_model:
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
            if "resnet18" in used_model and "cifar" not in used_model:
                bb = self.net(x)
                b, h, _, _ = bb.shape
                logits = self.t1_head(bb.view(b, h))
            else:
                logits = self.t1_head(self.net(x))
        return logits

class SeparateMTL(nn.Module):
    def __init__(self, task_keys):
        super().__init__()
        self.backbones = nn.ModuleDict({
            "t0": backbone_separate,
            "t1": backbone2_separate
        })

        if used_model == "cifarcresnet18":
            lin = nn.Linear(10, 10)
        else:
            lin = nn.Linear(512, 10)
        self.heads = nn.ModuleDict({
            task: lin
            for task in task_keys
        })
    
    def forward(self, x, task):
        if task == "t0":
            if t0_permute:
                x = torch.permute(x, (0, 3, 1, 2)).float()
        if task == "t1":
            if t1_permute:
                x = torch.permute(x, (0, 3, 1, 2)).float()

        if "resnet18" in used_model and "cifar" not in used_model:
            bb = self.backbones[task](x)
            b, h, _, _ = bb.shape
            logits = self.heads[task](bb.view((b, h)))
        else:
            logits = self.heads[task](self.backbones[task](x))
        return logits

shared_net = Shared_MTL()
shared_net.to(DEVICE)

separate_mtl = SeparateMTL(tasks.keys())
separate_mtl.to(DEVICE)

param_keys = []

# for k, v in shared_net.named_parameters():
    # if "head" in k:
    #     break
    # if "5" not in k and "6" not in k and "7" not in k:
    # print(k)
#     param_keys.append(k[4:])
# s = ""
# for p in param_keys:
#     s += p + ","
# print(s)
if positive:
    param_keys = np.loadtxt(path + "/param_keys_lambda.txt",
                 delimiter=",", dtype=str).tolist()
else:
    param_keys = np.loadtxt(path + "/param_keys.txt",
                    delimiter=",", dtype=str).tolist()
print(param_keys)

class SurgicalMTL(nn.Module):
    def __init__(self, task_keys):
        super().__init__()
        self.backbones = model.Shareable(
            mdl=backbone_surgical,
            task_keys=list(task_keys),
            shared_params=param_keys,
            used_model=used_model,
            pretrained=pretrained,
            mixer_depth=mixer_depth
        )

        if used_model == "cifarcresnet18":
            lin = nn.Linear(10, 10)
        else:
            lin = nn.Linear(512, 10)

        self.heads = nn.ModuleDict({
            task: lin
            for task in task_keys
        })
    
    def forward(self, x, task):
        if task == "t0":
            if t0_permute:
                x = torch.permute(x, (0, 3, 1, 2)).float()
        if task == "t1":
            if t1_permute:
                x = torch.permute(x, (0, 3, 1, 2)).float()

        if "resnet18" in used_model and "cifar" not in used_model:
            bb = self.backbones(x, task)
            b, h, _, _ = bb.shape
            logits = self.heads[task](bb.view((b, h)))
        else:
            logits = self.heads[task](self.backbones(x, task))
        return logits


mtl = SurgicalMTL(tasks.keys())
mtl.to(DEVICE)

fully_shared_exp = train.train_and_evaluate(
    model=shared_net,
    tasks=tasks,
    steps=n_step,
    lr=lr,
    eval_every=50,
    DEVICE=DEVICE
)

separate_exp = train.train_and_evaluate(
    model=separate_mtl,
    tasks=tasks,
    steps=n_step,
    lr=lr,
    eval_every=50,
    DEVICE=DEVICE
)

surgical_exp = train.train_and_evaluate(
    model=mtl,
    tasks=tasks,
    steps=n_step,
    lr=lr,
    eval_every=50,
    DEVICE=DEVICE
)

for task_name in tasks:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))

    lamb = ""
    if positive:
        lamb = "_lambda_"
    for exp_name, exp in [
        ('shared', fully_shared_exp),
        ('separate', separate_exp),
        ('surgical', surgical_exp),
    ]:
        losses, metrics, eval_losses, eval_metrics = exp
        
        tl = [(s, l[task_name]) for s, l in losses]
        el = [(s, l[task_name]) for s, l in eval_losses]
        tm = [(s, m[task_name]) for s, m in metrics]
        em = [(s, m[task_name]) for s, m in eval_metrics]
    
        avg = np.array([e for _, e in em])
        avgl = np.array([e for _, e in el])

        
        np.savetxt(path + "/" + task_name + exp_name + lamb + 'acc.txt', avg, delimiter=',')
        np.savetxt(path + "/" + task_name + exp_name + lamb + 'loss.txt', avgl, delimiter=',')
        # print(task_name + exp_name)
        # print("Last Acc:")
        # print(avg)

        # print("Last LOSS: ")
        # print(avgl)
        # plot
        # ax = axes[0]
        # tl_x, tl_y = zip(*tl)
        # ax.plot(tl_x, tl_y, label=f'{exp_name}_{task_name}')
        # ax.set_title('Train Loss')
        # ax.set_yscale('log')
        # ax.legend()

        ax = axes[0]
        el_x, el_y = zip(*el)
        ax.plot(el_x, el_y, label=f'{exp_name}_{task_name}')
        ax.set_title('Val Loss')
        ax.set_yscale('log')
        ax.legend()

        # ax = axes[2]
        # tm_x, tm_y = zip(*tm)
        # ax.plot(tm_x, tm_y, label=f'{exp_name}_{task_name}')
        # ax.set_title('Train Accuracy')
        # # ax.set_ylim([0.20, 1.])
        # ax.legend()

        ax = axes[1]
        em_x, em_y = zip(*em)
        ax.plot(em_x, em_y, label=f'{exp_name}_{task_name}')
        ax.set_title('Val Accuracy')
        # ax.set_ylim([0.20, 1.])
        ax.legend()
    
    fig.savefig(path + "/" + task_name + lamb + ".png")
