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

Usage: python src/scale_robustness.py
"""

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
from utils import get_device  # noqa: E402

MODEL_NAME = "YOLO26N_CLS"
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


def main() -> None:
    config = yaml.safe_load(open(BASE_DIR / "configs" / "config.yaml"))
    models_cfg = yaml.safe_load(open(BASE_DIR / "configs" / "models.yaml"))
    cfg = next(m for m in models_cfg["yolo_models"] if m["name"] == MODEL_NAME)

    ckpt = BASE_DIR / "log" / MODEL_NAME / "checkpoints" / "best_model.pt"
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    from ultralytics import YOLO
    model = YOLO(str(ckpt))
    get_device()

    _, _, test_loader = get_loaders(
        BASE_DIR / config["paths"]["data_dir"], cfg["batch_size"],
        cfg["input_size"][0], config["training"]["num_workers"])

    results = {}
    for s in SCALES:
        y_true, y_pred = [], []
        for inputs, targets in test_loader:
            raw = (inputs * _STD + _MEAN).clamp(0.0, 1.0)
            raw = rescale_in_frame(raw, s)
            for r, t in zip(model.predict(raw, verbose=False, task="classify"),
                            targets.numpy()):
                y_pred.append(int(np.argmax(r.probs.data.cpu().numpy())))
                y_true.append(int(t))
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = float((y_true == y_pred).mean())
        per_class = {int(c): float((y_pred[y_true == c] == c).mean())
                     for c in np.unique(y_true)}
        results[s] = {"accuracy": acc, "per_class_recall": per_class}
        print(f"  scale {s:.2f}  accuracy {acc:.4f}  "
              f"(delta {acc - results[1.00]['accuracy']:+.4f})"
              if 1.00 in results else f"  scale {s:.2f}  accuracy {acc:.4f}")

    out = BASE_DIR / "log" / "global_results" / "scale_robustness.json"
    json.dump({"model": MODEL_NAME, "classes": config["project"]["classes"],
               "results": {str(k): v for k, v in results.items()}},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
