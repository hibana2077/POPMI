#!/usr/bin/env python3
from __future__ import annotations
import argparse
import torch
from popmi.datasets.medmnist import medmnist_loaders
from popmi.engine.train import run_training

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='pneumoniamnist')
    p.add_argument('--arch', type=str, default='resnet18')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--image_size', type=int, default=64)
    p.add_argument('--as_rgb', action='store_true')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ce', action='store_true')
    p.add_argument('--popmi_beta', type=float, default=0.5)
    p.add_argument('--popmi_w', action='store_true')
    p.add_argument('--popmi_k', type=int, default=0)
    p.add_argument('--ce_iters', type=int, default=3)
    p.add_argument('--ce_rho', type=float, default=1.0)
    args = p.parse_args()
    train_loader, val_loader, test_loader, n_classes = medmnist_loaders(args.dataset, image_size=args.image_size, as_rgb=args.as_rgb, batch_size=args.batch_size)
    run_training(arch=args.arch, train_loader=train_loader, val_loader=val_loader, num_classes=n_classes, epochs=args.epochs,
                 lr=args.lr, device=args.device, use_ce=args.ce, popmi_beta=args.popmi_beta, popmi_w=args.popmi_w,
                 popmi_k=args.popmi_k, ce_iters=args.ce_iters, ce_rho=args.ce_rho, task='binary' if n_classes==2 else 'multiclass')
