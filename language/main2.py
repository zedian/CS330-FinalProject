import numpy as np
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, Dataset
# from cifar10_models.resnet import resnet18
from sklearn.model_selection import train_test_split
from robustbench.utils import load_model
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

accuracy = Accuracy()
corruptions=["gaussian_noise"]
# corruptions = ['frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow',
#                     'spatter', 'speckle_noise', 'zoom_blur']
cifar_c_x, cifar_c_y = load_cifar10c(n_examples=10000, corruptions=corruptions, severity=5)

cifar_cx_train, cifar_cx_test, cifar_cy_train, cifar_cy_test = train_test_split(cifar_c_x, 
                                                                                    cifar_c_y, test_size=0.30, 
                                                                                    random_state=42)

# cifar_fx_train, cifar_fx_test, cifar_fy_train, cifar_fy_test = train_test_split(cifar_f.data, 
#                                                                                     cifar_f.targets, test_size=0.30, 
#                                                                                     random_state=42)
cifar_fy_train = [0]*len(cifar_cy_train)
cifar_fy_test = [0]*len(cifar_cy_test)

for i in range(len(cifar_cy_train)):
    cifar_fy_train[i] = 9 - cifar_cy_train[i]

for i in range(len(cifar_cy_test)):
    cifar_fy_test[i] = 9 - cifar_cy_test[i]


class MultiTaskDataset:
    def __init__(self, x, t0_y, t1_y):
        self.x = x
        self.t0_y = t0_y
        self.t1_y = t1_y
        
    def __getitem__(self, idx):
        return self.x[idx], self.t0_y[idx], self.t1_y[idx]
            
    def __len__(self):
        return len(self.x)

X_train = MultiTaskDataset(cifar_cx_train, cifar_cy_train, cifar_fy_train)
X_test = MultiTaskDataset(cifar_cx_test, cifar_cy_test, cifar_fy_test)

trainloader = DataLoader(X_train, batch_size=256, shuffle=True)
testloader = DataLoader(X_test, batch_size=256, shuffle=False)

class MTL_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet_backbone = load_model("Sehwag2021Proxy_R18", dataset='cifar10', threat_model='Linf')
        # print(self.resnet_backbone)
        # self.resnet_backbone = nn.Sequential(*list(self.resnet_backbone.children())[:-1])
        self.t0_head = nn.Linear(10, 10)
        self.t1_head = nn.Linear(10, 10)
    
    def forward(self, x):
        t0_resnet = self.resnet_backbone(x)
        t1_resnet = self.resnet_backbone(x)
        t0_logits = self.t0_head(t0_resnet)
        t1_logits = self.t1_head(t1_resnet)
        # return t0_logits
        return t0_logits, t1_logits
    
t0grad = {}
t1grad = {}

def train_simple(
    *, 
    train_loader, 
    test_loader, 
    t0_weight=0.5,
    t1_weight=0.5,
    rgn=0,
    lr=1e-3, 
    epochs=5, 
):
    val_metrics = {'loss/total': [], 
                     'loss/t0': [], 
                     'loss/t1': [],
                     'acc/t0': [],
                     'acc/t1': [],
                    }
    model = MTL_resnet()
    t0_model = MTL_resnet()
    t1_model = MTL_resnet()
    model.to(DEVICE)
    t0_model.to(DEVICE)
    t1_model.to(DEVICE)

    first_conv = []
    last_conv = []
    shared_lr = []
    for k, v in model.named_parameters():
        if "layer1.0" in k and "conv1" in k:
            first_conv.append(v)
        elif "layer4.1" in k and "conv2" in k:
            last_conv.append(v)
        else:
            shared_lr.append(v)
    parameters = [{"params": first_conv}, {"params": last_conv}, {"params": shared_lr}]

    optimizer = torch.optim.Adam(parameters, lr=lr)
    t0_optimizer = torch.optim.Adam(optimizer.param_groups, lr = lr)
    t1_optimizer = torch.optim.Adam(optimizer.param_groups, lr = lr)
    
    for epoch in range(epochs):
        model.train()
        last_it = len(train_loader) - rgn
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):    
            # t0_x, t0_labels = batch
            x, t0_labels, t1_labels = batch

            # t0_x, t0_labels= t0_x.to(DEVICE), t0_labels.to(DEVICE)
            x, t0_labels, t1_labels = x.to(DEVICE), t0_labels.to(DEVICE), t1_labels.to(DEVICE)

            # t0_x = torch.permute(t0_x, (0, 3, 1, 2)).float()
            # t1_x = torch.permute(t1_x, (0, 3, 1, 2)).float()
            # t0_logits = model.forward(t0_x)
            t0_logits, t1_logits = model.forward(x)

            t0_loss = nn.functional.cross_entropy(t0_logits, t0_labels)
            t1_loss = nn.functional.cross_entropy(t1_logits, t1_labels)
            total_loss = t0_weight*t0_loss + t1_weight*t1_loss

            # t0_loss.backward(retain_graph=True)
            total_loss.backward(retain_graph=True)
            t0rgn = 1
            t1rgn = 1
            if i == last_it:
                t0_model.load_state_dict(model.state_dict())
                t0_optimizer.load_state_dict(optimizer.state_dict())
                t1_model.load_state_dict(model.state_dict())
                t1_optimizer.load_state_dict(optimizer.state_dict())
            if i >= last_it:
                for g, h in zip(t0_optimizer.param_groups, t1_optimizer.param_groups):
                    t0grad = torch.autograd.grad(t0_loss, g["params"][0], allow_unused=True, retain_graph=True)
                    t1grad = torch.autograd.grad(t1_loss, h["params"][0], allow_unused=True, retain_graph=True)
                    
                    if t0grad[0] is not None:
                        t0rgn = torch.linalg.norm(t0grad[0]/torch.linalg.norm(g["params"][0]))
                    if t1grad[0] is not None:
                        t1rgn = torch.linalg.norm(t1grad[0]/torch.linalg.norm(h["params"][0]))
                    # t0rgn, t1rgn = nn.functional.softmax(torch.Tensor([t0rgn, t1rgn]))
                    # print(t0rgn)
                    # print(t1rgn)
                    g["lr"] = t0rgn*g["lr"]
                    h["lr"] = t1rgn*h["lr"]
                    
                    t0rgn = 1
                    t1rgn = 1

                t0_optimizer.step()
                t1_optimizer.step()
                
                t0_optimizer.zero_grad()
                t1_optimizer.zero_grad()
            else:
                for k, v in model.named_parameters():
                    g = torch.autograd.grad(t0_loss)
                    if k in t0grad:
                        t0grad[k].append(g)
                    else:
                        t0grad[k] = [g]
                    if k in t1grad:
                        t1grad[k].append(g)
                    else:
                        t1grad[k] = [g]
                optimizer.step()
                optimizer.zero_grad()

        t0_val_loss = 0
        t0_val_acc = 0
        t1_val_loss = 0
        t1_val_acc = 0
        val_count = 0
        model.eval()
        for batch in test_loader:
            # t0_x, t0_labels = batch
            # t0_x, t0_labels = t0_x.to(DEVICE), t0_labels.to(DEVICE)
            x, t0_labels, t1_labels = batch
            # t0_x, t0_labels = t0_x.to(DEVICE), t0_labels.to(DEVICE)
            x, t0_labels, t1_labels = x.to(DEVICE), t0_labels.to(DEVICE), t1_labels.to(DEVICE)

            # t0_logits = model.forward(t0_x)
            t0_logits, t1_logits = model.forward(x)

            # t0_loss = nn.functional.cross_entropy(t0_logits, t0_labels)
            # t1_loss = nn.functional.cross_entropy(t1_logits, t1_labels)

            val_count += len(x)
            # t0_val_loss += t0_loss.item() * len(t0_x)

            t0_lgt = t0_logits.clone().detach().cpu()
            t1_lgt = t1_logits.clone().detach().cpu()

            t0_lbl = t0_labels.clone().detach().cpu()
            t1_lbl = t1_labels.clone().detach().cpu()

            t0_val_acc += accuracy(t0_lgt, t0_lbl) * len(x)
            t1_val_acc += accuracy(t1_lgt, t1_lbl) * len(x)

            # t0_val_acc += clean_accuracy(model, t0_x, t0_lbl) * len(t0_x)
            # t1_val_acc += clean_accuracy(model, t1_x, t1_lbl) * len(t1_x)


        val_metrics['acc/t0'].append(t0_val_acc.item() / val_count)
        val_metrics['acc/t1'].append(t1_val_acc.item() / val_count)

    return val_metrics

# fine_tune = train_simple(
#     train_loader=trainloader, 
#     test_loader=testloader,
#     t0_weight = 0,
#     t1_weight = 1,
#     lr=1e-3, 
#     epochs=15,
# )


# rgn_shared_metrics = train_simple(
#     train_loader=trainloader, 
#     test_loader=testloader,
#     t0_weight = 0.5,
#     t1_weight = 0.5,
#     rgn = 16,
#     lr=1e-3,
#     epochs=15,
# )

shared_metrics = train_simple(
    train_loader=trainloader, 
    test_loader=testloader,
    t0_weight = 0.5,
    t1_weight = 0.5,
    rgn = 0,
    lr=1e-3,
    epochs=1,
)

for k in t0grad.keys:
    cosine_dist = torch.sum(torch.functional.normalize(t0grad[k], dim=-1) * torch.functional.normalize(t1grad[k], dim=-1), dim=-1)
    print(cosine_dist)

print(shared_metrics)
# print(rgn_shared_metrics)

epochs = np.arange(1, 16, 1)
plt.figure(figsize=(12, 8))
plt.title("RGN for 15 batches vs no RGN")
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.plot(epochs, shared_metrics["acc/t0"], label="CIFAR10-C")
plt.plot(epochs, shared_metrics["acc/t1"], label="CIFAR10-C Flipped")
plt.plot(epochs, rgn_shared_metrics["acc/t0"], label="CIFAR10-C RGN")
plt.plot(epochs, rgn_shared_metrics["acc/t1"], label="CIFAR10-C Flipped RGN")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("0.5-0.5-cifar10-c_RGN.png", bbox_inches='tight')
# np.savetxt("./metrics.txt", shared_metrics)