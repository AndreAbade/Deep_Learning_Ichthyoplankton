"""Scale-robustness probe for the trained classifier.

Magnification was adjusted to specimen size during acquisition and therefore
varies among classes, so apparent size is a cue that correlates with the label.
A model that had learned apparent scale rather than morphology would lose
accuracy sharply when the specimen is rescaled inside the frame; a model reading
morphology should degrade gently and symmetrically.

This probe rescales each test image inside its 224x224 frame by a factor s and
re-evaluates the already-trained checkpoint. Nothing is retrained and no new
data is collected. Background is filled from each image's own border median so
that rescaling does not introduce a spurious uniform-colour cue.

Usage: python src/scale_robustness.py [--model NAME]

With no --model the probe follows whichever model heads ranking_models.csv, so
it always describes the model the paper reports as leading.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dataset import get_loaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device  # noqa: E402

SCALES = [0.60, 0.70, 0.85, 1.00, 1.20, 1.40]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def rescale_in_frame(batch: torch.Tensor, s: float) -> torch.Tensor:
    """Resize the content by s, keeping the output frame at the original size."""
    if abs(s - 1.0) < 1e-6:
        return batch
    n, c, h, w = batch.shape
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    resized = F.interpolate(batch, size=(nh, nw), mode="bilinear",
                            align_corners=False, antialias=True)
    if s < 1.0:  # shrink: paste centred on a background-coloured canvas
        border = torch.cat([batch[:, :, :2, :].reshape(n, c, -1),
                            batch[:, :, -2:, :].reshape(n, c, -1),
                            batch[:, :, :, :2].reshape(n, c, -1),
                            batch[:, :, :, -2:].reshape(n, c, -1)], dim=2)
        fill = border.median(dim=2).values.view(n, c, 1, 1)
        out = fill.expand(n, c, h, w).clone()
        top, left = (h - nh) // 2, (w - nw) // 2
        out[:, :, top:top + nh, left:left + nw] = resized
        return out
    top, left = (nh - h) // 2, (nw - w) // 2  # enlarge: centre-crop back
    return resized[:, :, top:top + h, left:left + w]


def _find_cfg(models_cfg: dict, name: str) -> dict:
    for group in ("cnn_models", "transformer_models", "yolo_models"):
        for m in models_cfg.get(group, []):
            if m["name"] == name:
                return m
    sys.exit(f"{name} is not declared in configs/models.yaml")


def _members(log_dir: Path, name: str) -> list[str]:
    """Ensemble_CNN has no checkpoint of its own; probe its members instead."""
    metrics = log_dir / name / "metrics" / "test_metrics.json"
    if metrics.exists():
        recorded = json.load(open(metrics)).get("ensemble_models")
        if recorded:
            return list(recorded)
    return [name]


def _probs_all_scales(cfg: dict, config: dict, log_dir: Path,
                      device) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """Class probabilities over the test set at every scale in SCALES.

    All scales are swept inside one pass over the loader: building it fills the
    RAM cache, so re-creating it per scale would dominate the runtime.
    """
    name, backend = cfg["name"], cfg["backend"]
    ckpt = log_dir / name / "checkpoints" / "best_model.pt"
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    _, _, test_loader = get_loaders(
        BASE_DIR / config["paths"]["data_dir"], cfg["batch_size"],
        cfg["input_size"][0], config["training"]["num_workers"])

    if backend == "ultralytics":
        from ultralytics import YOLO
        model = YOLO(str(ckpt))
    else:
        model = build_model(name, backend, cfg["key"],
                            num_classes=config["project"]["num_classes"])
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device).eval()

    probs = {s: [] for s in SCALES}
    labels = []
    for inputs, targets in test_loader:
        # Rescaling happens in un-normalised space so the border-median fill is
        # an actual image colour rather than a normalised artefact.
        raw_unit = (inputs * _STD + _MEAN).clamp(0.0, 1.0)
        for s in SCALES:
            raw = rescale_in_frame(raw_unit, s)
            if backend == "ultralytics":
                for r in model.predict(raw, verbose=False, task="classify"):
                    probs[s].append(r.probs.data.cpu().numpy())
            else:
                with torch.no_grad():
                    out = model(((raw - _MEAN) / _STD).to(device))
                    if isinstance(out, tuple):
                        out = out[0]
                    probs[s].extend(torch.softmax(out, dim=1).cpu().numpy())
        labels.extend(targets.numpy().tolist())
    return {s: np.asarray(v) for s, v in probs.items()}, np.asarray(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="model to probe; defaults to the top of ranking_models.csv")
    args = ap.parse_args()

    config = yaml.safe_load(open(BASE_DIR / "configs" / "config.yaml"))
    models_cfg = yaml.safe_load(open(BASE_DIR / "configs" / "models.yaml"))
    log_dir = BASE_DIR / "log"

    model_name = args.model
    if model_name is None:
        import pandas as pd
        ranking = log_dir / "global_results" / "ranking_models.csv"
        if not ranking.exists():
            sys.exit(f"no --model given and {ranking} does not exist")
        model_name = pd.read_csv(ranking).iloc[0]["model"]
        print(f"probing the leading model: {model_name}")

    members = _members(log_dir, model_name)
    if members != [model_name]:
        print(f"{model_name} is an ensemble of {' + '.join(members)}; "
              f"averaging their probabilities at each scale")
    device = get_device()

    summed, y_true = None, None
    for member in members:
        print(f"  [{member}] sweeping scales...")
        p, labels = _probs_all_scales(_find_cfg(models_cfg, member), config,
                                      log_dir, device)
        summed = p if summed is None else {s: summed[s] + p[s] for s in SCALES}
        y_true = labels

    results = {}
    for s in SCALES:
        y_pred = summed[s].argmax(axis=1)
        acc = float((y_true == y_pred).mean())
        per_class = {int(c): float((y_pred[y_true == c] == c).mean())
                     for c in np.unique(y_true)}
        results[s] = {"accuracy": acc, "per_class_recall": per_class}
        delta = f"  (delta {acc - results[1.00]['accuracy']:+.4f})" if 1.00 in results else ""
        print(f"  scale {s:.2f}  accuracy {acc:.4f}{delta}")

    out = log_dir / "global_results" / "scale_robustness.json"
    json.dump({"model": model_name, "members": members,
               "classes": config["project"]["classes"],
               "results": {str(k): v for k, v in results.items()}},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
