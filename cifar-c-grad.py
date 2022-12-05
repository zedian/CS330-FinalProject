import itertools

import matplotlib.pyplot as plt
import torch

import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from robustbench.utils import load_model
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy
from torchmetrics import Accuracy
from cifar10_models.resnet import resnet18
from mlp_mixer_pytorch import MLPMixer

import dataset
import model
import utils
import train

batch_size = 256
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

accuracy = Accuracy().to(DEVICE)
# accuracy = Accuracy()
corruptions=["gaussian_noise"]

cifarc_train = dataset.CIFAR10C(corruptions, flipped=False, ratio=0.2)
cifarc_flipped_train = dataset.CIFAR10C(corruptions, flipped=True, ratio=0.2)

cifarc_valid = dataset.CIFAR10C(corruptions, flipped=False, train=False, ratio=0.2)
cifarc_flipped_valid = dataset.CIFAR10C(corruptions, flipped=True, train=False, ratio=0.2)

cifarc_train_loader = DataLoader(cifarc_train, batch_size=batch_size, shuffle=True, drop_last=True)
cifarc_flipped_train_loader = DataLoader(cifarc_flipped_train, batch_size=batch_size, shuffle=True, drop_last=True)

cifarc_valid_loader = DataLoader(cifarc_valid, batch_size=batch_size, shuffle=True, drop_last=True)
cifarc_flipped_valid_loader = DataLoader(cifarc_flipped_valid, batch_size=batch_size, shuffle=True, drop_last=True)
cifarc_iter = itertools.cycle(cifarc_train_loader)
cifarc_flipped_iter = itertools.cycle(cifarc_flipped_train_loader)

tasks = {
    't0': {
        'train_iter': cifarc_iter,
        'eval_ds': cifarc_valid_loader,
        'loss': lambda logits, labels: F.cross_entropy(logits, labels),
        'predict': lambda logits: logits,
        'metric': lambda yh, y: accuracy(yh, y),
    },
    't1': {
        'train_iter': cifarc_flipped_iter,
        'eval_ds': cifarc_flipped_valid_loader,
        'loss': lambda logits, labels: F.cross_entropy(logits, labels),
        'predict': lambda logits: logits,
        'metric': lambda yh, y: accuracy(yh, y),
    },
}

# class SharedMTL(nn.Module):
#     def __init__(self, task_keys):
#         super().__init__()
#         self.backbone = model.LinearBackbone()
#         self.heads = nn.ModuleDict({
#             task: nn.Linear(256, 1)
#             for task in task_keys
#         })
    
#     def forward(self, x, task):
#         return self.heads[task](self.backbone(x)) 

class MTL_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18()
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        # mdl = MLPMixer(
        #         image_size = 32,
        #         channels = 3,
        #         patch_size = 8,
        #         dim = 512,
        #         depth = 12,
        #         num_classes = 10
        #     )
        # self.net = mdl
        # self.net = load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf')
        # print(self.resnet_backbone)
        # self.resnet_backbone = nn.Sequential(*list(self.resnet_backbone.children())[:-1])
        self.t0_head = nn.Linear(512, 10)
        self.t1_head = nn.Linear(512, 10)
    
    def forward(self, x, task):
        # print(task)
        x = torch.permute(x, (0, 3, 1, 2)).float()
        bb = self.net(x)
        b, h, _, _ = bb.shape
        if task == "t0":
            # logits = self.t0_head(self.net(x))
            logits = self.t0_head(bb.view(b, h))
        if task == "t1":
            # logits = self.t1_head(self.net(x))
            logits = self.t1_head(bb.view(b, h))
        # return t0_logits
        return logits

# shared_mtl = SharedMTL(tasks.keys())
shared_resnet = MTL_resnet()
shared_resnet.to(DEVICE)

param_keys = []
# for k, v in shared_resnet.named_parameters():
#     # print(k)
#     # if "conv1" in k:
#     if "head" in k: 
#         break
#     param_keys.append(k[4:])


fully_shared_exp = train.train_and_evaluate(
    model=shared_resnet,
    tasks=tasks,
    steps=2500,
    lr=1e-4,
    eval_every=50,
    DEVICE=DEVICE
)

class SeparateMTL(nn.Module):
    def __init__(self, task_keys):
        super().__init__()
        # mdl = MLPMixer(
        #         image_size = 32,
        #         channels = 3,
        #         patch_size = 8,
        #         dim = 512,
        #         depth = 12,
        #         num_classes = 10
        #     )
        # self.backbones = nn.ModuleDict({
        #     task: mdl for task in task_keys
        # })
        # self.backbones = nn.ModuleDict({
        #     task: load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf')
        #     for task in task_keys
        # })
        mdl1 = resnet18()
        mdl1 = nn.Sequential(*list(mdl1.children())[:-1])
        mdl2 = resnet18()
        mdl2 = nn.Sequential(*list(mdl2.children())[:-1])
        self.backbones = nn.ModuleDict({
            "t0": mdl1,
            "t1": mdl2
        })
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10)
            for task in task_keys
        })
    
    def forward(self, x, task):
        x = torch.permute(x, (0, 3, 1, 2)).float()
        bb = self.backbones[task](x)
        b, h, _, _ = bb.shape
        # return self.heads[task](self.backbones[task](x))
        return self.heads[task](bb.view((b, h)))

separate_mtl = SeparateMTL(tasks.keys())
separate_mtl.to(DEVICE)
separate_exp = train.train_and_evaluate(
    model=separate_mtl,
    tasks=tasks,
    steps=2500,
    lr=1e-4,
    eval_every=50,
    DEVICE=DEVICE
)


# del shared_resnet
# gc.collect()
# for k, v in shared_resnet.named_parameters():
#     # print(k)
#     # if "head" in k: 
#     #     break
#     if "conv1" in k:
#         param_keys.append(k)

for k, v in shared_resnet.named_parameters():
    if "head" in k: 
        break
    if "7" not in k and "conv" not in k:
        param_keys.append(k[4:]) 
    
# for k, v in shared_resnet.named_parameters():
#     print(k)
#     if "head" in k: 
#         break
#     if "conv" not in k and "layer4" not in k:
#         # print(k[4:])
#         param_keys.append(k[4:])

# for k, v in shared_resnet.named_parameters():
#     # print(k)
#     if "shortcut" in k:
#         param_keys.append(k)
# for k, v in shared_resnet.named_parameters():
#     # print(k)
#     if "linear" in k:
#         param_keys.append(k)
#     if "layer1" in k and "conv1" in k:
#         param_keys.append(k)
#     if "layer4" in k and "conv2" in k:
#         param_keys.append(k)

# for k, v in shared_resnet.named_parameters():
#     print(k)
#     if "layer2" in k and "conv1" in k:
#         param_keys.append(k)
#     if "layer3" in k and "conv1" in k:
#         param_keys.append(k)
#     if "layer3" in k and "conv2" in k:
#         param_keys.append(k)

# for k, v in shared_resnet.named_parameters():
#     if "head" in k:
#         break
#     if "16.weight" not in k:
#         param_keys.append(k[4:])
# for k, v in shared_resnet.named_parameters():
#     if "16.weight" in k:
#         param_keys.append(k)
#     if "14.weight" in k:
#         param_keys.append(k)
#     if "net.13." in k and "fn" in k and "weight" in k:
#         param_keys.append(k)
#     if "net.3." in k and "fn" in k and "weight" in k:
#         param_keys.append(k)
# param_keys = ['backbone.' + k for k in list(shared_mtl.backbone.state_dict().keys())]
# print(param_keys)

# print(param_keys)
# grads = train.get_gradients(
#     model=shared_resnet,
#     tasks=tasks, 
#     steps= 500, 
#     lr=1e-4,
#     DEVICE=DEVICE,
#     param_keys=param_keys
# )

# # # del shared_resnet
# heuristic_results = {}

# # plots
# n_rows = 3
# n_cols = 3
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

# for i, key in enumerate(param_keys):
#     # get gradients
#     # g0 = utils.stack_grad(grads, 't0', key)
#     # g1 = utils.stack_grad(grads, 't1', key)
    
#     # heuristics computations
#     cosine = torch.sum(F.normalize(utils.stack_grad(grads, 't1', key), dim=-1) * F.normalize(utils.stack_grad(grads, 't0', key), dim=-1), dim=-1)
#     smooth_cos = utils.low_pass_filter(cosine[None], filter_size=10)[0][0]
    
#     avg_smoothed_cos = smooth_cos.mean()
#     std_cos = cosine.std()
    
#     heuristic_results[key] = {'avg_cos': avg_smoothed_cos, 'std_cos': std_cos}
      
#     # plot
#     row, col = i // n_cols, i % n_cols
#     ax = axes[row][col]    
#     ax.set_title(key)
#     if col == 0:
#         ax.set_ylabel('Gradient cosine similarity')
#     if row == n_rows - 1:
#         ax.set_xlabel('Step')
#     ax.set_ylim([-1.1, 1.1])
#     ax.plot(cosine, color='teal', alpha=0.2)
#     ax.plot(smooth_cos, color='teal')

# fig.savefig("fig_resnet_shortcut.png")

# print(heuristic_results)
class SurgicalMTL(nn.Module):
    def __init__(self, task_keys):
        super().__init__()
        # mdl = MLPMixer(
        #         image_size = 32,
        #         channels = 3,
        #         patch_size = 8,
        #         dim = 512,
        #         depth = 12,
        #         num_classes = 10
        #     )
        # self.backbone = model.Shareable(
        #     mdl=load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf'),
        #     task_keys=list(task_keys),
        #     shared_params=param_keys
        # )
        # self.backbone = model.Shareable(
        #     mdl=mdl,
        #     task_keys=list(task_keys),
        #     shared_params=param_keys
        # )
        mdl1 = resnet18()
        mdl1 = nn.Sequential(*list(mdl1.children())[:-1])
        self.backbone = model.Shareable(
            mdl=mdl1,
            task_keys=list(task_keys),
            shared_params=param_keys
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10)
            for task in task_keys
        })
    
    def forward(self, x, task):
        x = torch.permute(x, (0, 3, 1, 2)).float()
        bb = self.backbone(x, task)
        b, h, _, _ = bb.shape
        # return self.heads[task](self.backbone(x, task))
        return self.heads[task](bb.view((b, h)))
mtl = SurgicalMTL(tasks.keys())
mtl.to(DEVICE)
# r = torch.cuda.memory_reserved(0)
# a = torch.cuda.memory_allocated(0)
# print(r-a)
surgical_exp = train.train_and_evaluate(
    model=mtl,
    tasks=tasks,
    steps=2500,
    lr=1e-4,
    eval_every=50,
    DEVICE=DEVICE
)


for task_name in tasks:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))

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
    
        # plot
        ax = axes[0]
        tl_x, tl_y = zip(*tl)
        ax.plot(tl_x, tl_y, label=f'{exp_name}_{task_name}')
        ax.set_title('Train Loss')
        ax.set_yscale('log')
        ax.legend()

        ax = axes[1]
        el_x, el_y = zip(*el)
        ax.plot(el_x, el_y, label=f'{exp_name}_{task_name}')
        ax.set_title('Val Loss')
        ax.set_yscale('log')
        ax.legend()

        ax = axes[2]
        tm_x, tm_y = zip(*tm)
        ax.plot(tm_x, tm_y, label=f'{exp_name}_{task_name}')
        ax.set_title('Train Accuracy')
        # ax.set_ylim([0.20, 1.])
        ax.legend()

        ax = axes[3]
        em_x, em_y = zip(*em)
        ax.plot(em_x, em_y, label=f'{exp_name}_{task_name}')
        ax.set_title('Val Accuracy')
        # ax.set_ylim([0.20, 1.])
        ax.legend()
    fig.savefig("resnet_surgical" + task_name + ".png")


# print(heuristic_results)
# cifar_c_x, cifar_c_y = load_cifar10c(n_examples=10000, corruptions=corruptions, severity=5)
# cifar_cx_train, cifar_cx_test, cifar_cy_train, cifar_cy_test = train_test_split(cifar_c_x, 
#                                                                                     cifar_c_y, test_size=0.30, 
#                                                                                     random_state=42)

# cifar_fy_train = [0]*len(cifar_cy_train)
# cifar_fy_test = [0]*len(cifar_cy_test)

# for i in range(len(cifar_cy_train)):
#     cifar_fy_train[i] = 9 - cifar_cy_train[i]

# for i in range(len(cifar_cy_test)):
#     cifar_fy_test[i] = 9 - cifar_cy_test[i]

# train_loader = DataLoader(cifar_, batch_size=bs, shuffle=True, drop_last=True)