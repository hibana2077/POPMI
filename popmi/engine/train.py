from __future__ import annotations
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Dict
from ..models.model import build_model
from ..metrics.classification import binary_auroc, multiclass_balanced_accuracy

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0; self.cnt = 0
    def update(self, v: float, n: int = 1):
        self.sum += v * n; self.cnt += n
    @property
    def avg(self):
        return self.sum / max(self.cnt,1)

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_meter.update(loss.item(), imgs.size(0))
    return {'train_loss': loss_meter.avg}

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, task: str = 'binary') -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss_meter.update(loss.item(), imgs.size(0))
            probs = torch.softmax(logits, dim=1)
            if task == 'binary':
                all_probs.append(probs[:,1].cpu())
            else:
                all_probs.append(probs.cpu())
            all_targets.append(labels.cpu())
    if task == 'binary':
        probs_cat = torch.cat(all_probs)
        targets_cat = torch.cat(all_targets)
        auroc = binary_auroc(probs_cat, targets_cat)
        preds = (probs_cat >= 0.5).long()
        acc = (preds == targets_cat).float().mean().item()
        return {'val_loss': loss_meter.avg, 'auroc': auroc, 'acc': acc}
    else:
        probs_cat = torch.cat(all_probs)
        targets_cat = torch.cat(all_targets)
        preds = probs_cat.argmax(dim=1)
        bacc = multiclass_balanced_accuracy(preds, targets_cat)
        return {'val_loss': loss_meter.avg, 'balanced_acc': bacc}

def run_training(arch: str, train_loader: DataLoader, val_loader: DataLoader, num_classes: int,
                 epochs: int = 10, lr: float = 3e-4, device: str = 'cuda', use_ce: bool = True,
                 popmi_beta: float = 0.5, popmi_w: bool = False, popmi_k: int = 0,
                 ce_iters: int = 3, ce_rho: float = 1.0, task: str = 'binary'):
    dev = torch.device(device)
    model = build_model(arch=arch, pretrained=False, in_chans=3, num_classes=num_classes,
                        use_ce=use_ce, popmi_beta=popmi_beta, popmi_learnable_w=popmi_w,
                        popmi_sample_k=(popmi_k if popmi_k>0 else None), ce_iters=ce_iters, ce_rho=ce_rho).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        ev = evaluate(model, val_loader, dev, task=task)
        print(f"Epoch {epoch}: {tr}|{ev}")
    return model
