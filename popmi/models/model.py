from __future__ import annotations
import timm
import torch
from torch import nn, Tensor
from .popmi_layers import POPMIProjection, CE_PnP
from typing import Optional

class POPMIWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: int,
                 use_ce: bool = True, popmi_beta: float = 0.5, popmi_learnable_w: bool = False,
                 popmi_sample_k: Optional[int] = None, ce_iters: int = 3, ce_rho: float = 1.0,
                 projector_position: str = 'prepool'):
        super().__init__()
        self.backbone = backbone
        self.projector_position = projector_position
        projector = POPMIProjection(beta=popmi_beta, learnable_weights=popmi_learnable_w,
                                    sample_k=popmi_sample_k)
        self.popmi = CE_PnP(projector, rho=ce_rho, iters=ce_iters) if use_ce else projector
        self.head = nn.Linear(feat_dim, n_classes)
    def forward_features(self, x: Tensor) -> Tensor:
        return self.backbone.forward_features(x)
    def forward(self, x: Tensor) -> Tensor:
        feats = self.forward_features(x)
        # Expect shape (B,C,H,W) before pooling for CNNs; some timm models output (B, C) directly.
        if feats.ndim == 2:  # already pooled (e.g., ViT cls token), skip spatial projection
            z = feats
        else:
            z_proj = self.popmi(feats)
            z = z_proj.mean(dim=(-2, -1))
        logits = self.head(z)
        return logits

def build_model(arch: str = 'resnet18', pretrained: bool = False, in_chans: int = 3,
                num_classes: int = 2, use_ce: bool = True, popmi_beta: float = 0.5,
                popmi_learnable_w: bool = False, popmi_sample_k: Optional[int] = None,
                ce_iters: int = 3, ce_rho: float = 1.0) -> nn.Module:
    m = timm.create_model(arch, pretrained=pretrained, in_chans=in_chans, num_classes=0)  # feature extractor
    # Determine feature dim: try get_classifier or num_features
    feat_dim = getattr(m, 'num_features', None)
    if feat_dim is None:
        # Fallback: run a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, 224, 224)
            f = m.forward_features(dummy)
            if f.ndim == 4:
                feat_dim = f.shape[1]
            else:
                feat_dim = f.shape[-1]
    model = POPMIWrapper(m, feat_dim=feat_dim, n_classes=num_classes,
                         use_ce=use_ce, popmi_beta=popmi_beta,
                         popmi_learnable_w=popmi_learnable_w, popmi_sample_k=popmi_sample_k,
                         ce_iters=ce_iters, ce_rho=ce_rho)
    return model
