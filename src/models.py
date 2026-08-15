from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 6

# ──────────────────────────────────────────────────────────────────────────────
# Torchvision builders
# ──────────────────────────────────────────────────────────────────────────────

def _replace_classifier(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """Replace the final classifier head with a fresh linear layer."""
    n = name.lower()

    if n.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif n.startswith("vgg"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)

    elif n.startswith("resnet") or n.startswith("resnext") or n.startswith("inception_v3"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif n.startswith("convnext"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif n.startswith("mobilenet_v2"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif n.startswith("mobilenet_v3"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif n.startswith("efficientnet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif n == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif n == "swin_t":
        model.head = nn.Linear(model.head.in_features, num_classes)

    else:
        raise ValueError(f"Unknown torchvision model key for classifier replacement: {name}")

    return model


def _build_torchvision(key: str, num_classes: int, pretrained: bool) -> nn.Module:
    import torchvision.models as tvm
    weights_arg = "DEFAULT" if pretrained else None
    if key == "inception_v3":
        # Pesos DEFAULT exigem aux_logits=True na construção; depois desativamos
        # manualmente o branch para evitar o crash em entradas 224×224
        # (a aux Conv2d 5×5 recebe mapa 3×3 e falha).
        model = tvm.inception_v3(weights=weights_arg, aux_logits=True)
        model.AuxLogits = None
        model.aux_logits = False
    else:
        model = getattr(tvm, key)(weights=weights_arg)
    return _replace_classifier(model, key, num_classes)


def _build_timm(key: str, num_classes: int, pretrained: bool) -> nn.Module:
    import timm
    if pretrained:
        try:
            return timm.create_model(key, pretrained=True, num_classes=num_classes)
        except RuntimeError as exc:
            if "No pretrained weights exist" in str(exc):
                print(f"[WARNING] No pretrained weights for '{key}' — training from scratch.")
                return timm.create_model(key, pretrained=False, num_classes=num_classes)
            raise
    return timm.create_model(key, pretrained=False, num_classes=num_classes)


# ──────────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    name: str,
    backend: str,
    key: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Return a ready-to-train PyTorch model with the head replaced.

    For YOLO models, returns None — training is handled by yolo_runner.py.
    """
    if backend == "ultralytics":
        return None  # handled separately

    if backend == "torchvision":
        return _build_torchvision(key, num_classes, pretrained)

    if backend == "timm":
        return _build_timm(key, num_classes, pretrained)

    raise ValueError(f"Unknown backend: {backend}")


# ──────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_classifier_module(model: nn.Module, backend: str, key: str) -> nn.Module:
    """Return the classifier head module for any supported architecture."""
    if backend == "timm":
        # timm exposes a standard API regardless of internal attribute name
        # (e.g. inception_resnet_v2 uses 'classif', nasnetalarge uses 'last_linear')
        return model.get_classifier()

    k = key.lower()
    if k.startswith("densenet"):
        return model.classifier
    if k.startswith("vgg"):
        return model.classifier[-1]
    if k.startswith("resnet") or k.startswith("resnext") or k == "inception_v3":
        return model.fc
    if k.startswith("mobilenet") or k.startswith("efficientnet") or k.startswith("convnext"):
        return model.classifier[-1]
    if k == "vit_b_16":
        return model.heads.head
    if k == "swin_t":
        return model.head
    raise ValueError(f"Cannot locate classifier head for backend={backend}, key={key}")


def freeze_backbone(model: nn.Module, backend: str, key: str) -> None:
    clf = _get_classifier_module(model, backend, key)
    clf_ids = {id(p) for p in clf.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in clf_ids


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def wrap_multigpu(model: nn.Module) -> nn.Module:
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model
