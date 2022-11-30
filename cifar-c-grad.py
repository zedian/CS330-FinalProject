import itertools

import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from robustbench.utils import load_model
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy

import datasets
import model
import utils
import train

batch_size = 256
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# accuracy = Accuracy()
corruptions=["gaussian_noise"]

cifarc_train = datasets.CIFAR10C(corruptions, flipped=False)
cifarc_flipped_train = datasets.CIFAR10C(corruptions, flipped=True)

cifarc_valid = datasets.CIFAR10C(corruptions, flipped=False, train=False)
cifarc_flipped_valid = datasets.CIFAR10C(corruptions, flipped=True, train=False)

cifarc_train_loader = DataLoader(cifarc_train, batch_size=batch_size, shuffle=True, drop_last=True)
cifarc_flipped_train_loader = DataLoader(cifarc_flipped_train, batch_size=batch_size, shuffle=True, drop_last=True)

cifarc_iter = itertools.cycle(cifarc_train_loader)
cifarc_flipped_iter = itertools.cycle(cifarc_flipped_train_loader)

tasks = {
    't0': {
        'train_iter': cifarc_iter,
        'eval_ds': cifarc_valid,
        'loss': lambda logits, labels: F.cross_entropy(logits, labels),
        'predict': lambda logits: torch.sigmoid(logits) > 0.5,
        'metric': lambda yh, y: utils.accuracy(yh, y.long().bool()[..., None]),
    },
    't1': {
        'train_iter': cifarc_flipped_iter,
        'eval_ds': cifarc_flipped_valid,
        'loss': lambda logits, labels: F.cross_entropy(logits, labels),
        'predict': lambda logits: torch.sigmoid(logits) > 0.5,
        'metric': lambda yh, y: utils.accuracy(yh, y.long().bool()[..., None]),
    },
}

class SharedMTL(nn.Module):
    def __init__(self, task_keys):
        super().__init__()
        self.backbone = model.LinearBackbone()
        self.heads = nn.ModuleDict({
            task: nn.Linear(256, 1)
            for task in task_keys
        })
    
    def forward(self, x, task):
        return self.heads[task](self.backbone(x)) 

class MTL_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet_backbone = load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf')
        # print(self.resnet_backbone)
        # self.resnet_backbone = nn.Sequential(*list(self.resnet_backbone.children())[:-1])
        self.t0_head = nn.Linear(10, 10)
        self.t1_head = nn.Linear(10, 10)
    
    def forward(self, x, task):
        # print(task)
        x = torch.permute(x, (0, 3, 1, 2)).float()
        resnet = self.resnet_backbone(x)
        # print(resnet.shape)
        if task == "t0":
            logits = self.t0_head(resnet)
        if task == "t1":
            logits = self.t1_head(resnet)
        # return t0_logits
        return logits

# shared_mtl = SharedMTL(tasks.keys())
shared_resnet = MTL_resnet()
shared_resnet.to(DEVICE)

param_keys = []
# for k, v in shared_resnet.named_parameters():
#     # print(k)
#     if "linear" in k:
#         param_keys.append(k)
#     if "layer1" in k and "conv1" in k:
#         param_keys.append(k)
#     if "layer4" in k and "conv2" in k:
#         param_keys.append(k)
        
for k, v in shared_resnet.named_parameters():
    # print(k)
    if "layer2" in k and "conv1" in k:
        param_keys.append(k)
    if "layer3" in k and "conv1" in k:
        param_keys.append(k)
    if "layer3" in k and "conv2" in k:
        param_keys.append(k)
        
# param_keys = ['backbone.' + k for k in list(shared_mtl.backbone.state_dict().keys())]
print(param_keys)

grads = train.get_gradients(
    model=shared_resnet,
    tasks=tasks, 
    steps=80, 
    lr=3e-4,
    DEVICE=DEVICE,
    param_keys=param_keys
)

# del shared_resnet
heuristic_results = {}

# plots
n_rows = 2
n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

for i, key in enumerate(param_keys):
    # get gradients
    # g0 = utils.stack_grad(grads, 't0', key)
    # g1 = utils.stack_grad(grads, 't1', key)
    
    # heuristics computations
    cosine = torch.sum(F.normalize(utils.stack_grad(grads, 't1', key), dim=-1) * F.normalize(utils.stack_grad(grads, 't0', key), dim=-1), dim=-1).clone().detach().cpu()
    smooth_cos = utils.low_pass_filter(cosine[None], filter_size=10)[0][0]
    
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

fig.savefig("fig.png")
print(heuristic_results)
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