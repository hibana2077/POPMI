#!/usr/bin/env python3
from __future__ import annotations
import argparse
import torch
from popmi.models.model import build_model
from popmi.utils.profiling import profile_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default='resnet18')
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--ce', action='store_true')
    ap.add_argument('--popmi_beta', type=float, default=0.5)
    ap.add_argument('--popmi_w', action='store_true')
    ap.add_argument('--popmi_k', type=int, default=0)
    ap.add_argument('--ce_iters', type=int, default=3)
    ap.add_argument('--ce_rho', type=float, default=1.0)
    ap.add_argument('--image_size', type=int, default=224)
    args = ap.parse_args()
    model = build_model(arch=args.arch, pretrained=False, in_chans=3, num_classes=args.num_classes,
                        use_ce=args.ce, popmi_beta=args.popmi_beta, popmi_learnable_w=args.popmi_w,
                        popmi_sample_k=(args.popmi_k if args.popmi_k>0 else None), ce_iters=args.ce_iters, ce_rho=args.ce_rho)
    stats = profile_model(model, input_size=(3,args.image_size,args.image_size), device=args.device)
    print(stats)

if __name__ == '__main__':
    main()
