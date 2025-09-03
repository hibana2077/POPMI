"""Compatibility wrapper: original monolithic POPMI demo now delegates to package modules.

Use scripts/train_med.py for MedMNIST experiments.
This file keeps a minimal FakeData demo for quick sanity (no local testing executed here per user instruction).
"""
from __future__ import annotations
import argparse
import torch
from torchvision import transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torch import nn

from popmi.models.model import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default='resnet18')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--image_size', type=int, default=64)
    ap.add_argument('--classes', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--ce', action='store_true')
    ap.add_argument('--popmi_beta', type=float, default=0.5)
    ap.add_argument('--popmi_w', action='store_true')
    ap.add_argument('--popmi_k', type=int, default=0)
    ap.add_argument('--ce_iters', type=int, default=3)
    ap.add_argument('--ce_rho', type=float, default=1.0)
    args = ap.parse_args()
    device = torch.device(args.device)
    T = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    ds = FakeData(size=1000, image_size=(3,args.image_size,args.image_size), num_classes=args.classes, transform=T)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = build_model(arch=args.arch, pretrained=False, in_chans=3, num_classes=args.classes,
                        use_ce=args.ce, popmi_beta=args.popmi_beta, popmi_learnable_w=args.popmi_w,
                        popmi_sample_k=(args.popmi_k if args.popmi_k>0 else None), ce_iters=args.ce_iters, ce_rho=args.ce_rho).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = crit(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 20 == 0:
                acc = (logits.argmax(1)==labels).float().mean().item()
                print(f"ep{epoch} step{step} loss{loss.item():.4f} acc{acc:.3f}")
            step += 1
            if step >= args.steps:
                break
    print('Demo finished.')

if __name__ == '__main__':
    main()
