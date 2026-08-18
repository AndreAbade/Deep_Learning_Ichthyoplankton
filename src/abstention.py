"""Selective-classification policy: calibrate and threshold on validation only.

The coverage curve in plots.py is retrospective. It ranks the test predictions
by confidence and finds where the first error falls, so the operating point is
chosen with knowledge of the labels it is meant to predict and is optimistic by
construction. It bounds what the confidence ordering can separate; it is not a
policy a deployment could adopt.

This module fits everything on validation and touches the test set once:

  1. temperature-scale each ensemble member by minimising validation NLL
  2. average the calibrated probabilities, as the ensemble itself is built
  3. pick the confidence threshold on validation to meet a target risk
  4. freeze it and apply it to the test set, reporting selective risk with a
     confidence interval, per-class breakdown, and calibration metrics

Temperature scaling is applied to log-probabilities rather than logits for the
test set, which is equivalent because softmax((z - c)/T) = softmax(z/T), and
lets the stored y_prob.npy files be reused without re-running inference.

Usage: python src/abstention.py [--targets 0 0.01 0.02 0.05]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from scipy.stats import beta

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dataset import get_loaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import ensure_dir, get_device, set_seed  # noqa: E402

N_BOOTSTRAP = 2000
ECE_BINS = 15


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(logits / T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    r = minimize_scalar(lambda T: _nll(logits, y, T), bounds=(0.05, 10.0), method="bounded")
    return float(r.x)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def calibration_report(p: np.ndarray, y: np.ndarray) -> dict:
    pred, conf = p.argmax(1), p.max(1)
    correct = pred == y
    return {
        "accuracy": float(correct.mean()),
        "ece": expected_calibration_error(conf, correct),
        "brier": brier_score(p, y),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()),
    }


def validation_logits(members: list[str], config: dict, models_cfg: dict) -> tuple[dict, np.ndarray]:
    """Run each member over the validation split. The pipeline stores test
    predictions only, and a policy cannot be fitted on the test set."""
    device = get_device()
    logits, y_val = {}, None
    for name in members:
        cfg = None
        for group in ("cnn_models", "transformer_models", "yolo_models"):
            for m in models_cfg.get(group, []):
                if m["name"] == name:
                    cfg = m
        if cfg is None:
            sys.exit(f"{name} is not declared in configs/models.yaml")

        ckpt = BASE_DIR / "log" / name / "checkpoints" / "best_model.pt"
        if not ckpt.exists():
            sys.exit(f"checkpoint not found: {ckpt}")

        _, val_loader, _ = get_loaders(
            BASE_DIR / config["paths"]["data_dir"], cfg["batch_size"],
            cfg["input_size"][0], config["training"]["num_workers"])

        model = build_model(name, cfg["backend"], cfg["key"],
                            num_classes=config["project"]["num_classes"])
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device).eval()

        chunks, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                o = model(x.to(device))
                if isinstance(o, tuple):
                    o = o[0]
                chunks.append(o.cpu().numpy())
                labels.append(y.numpy())
        logits[name] = np.concatenate(chunks)
        y_val = np.concatenate(labels)
        print(f"  [{name}] validation logits {logits[name].shape}")
    return logits, y_val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05],
                    help="target selective risk values for which to pick a threshold")
    args = ap.parse_args()

    config = yaml.safe_load(open(BASE_DIR / "configs" / "config.yaml"))
    models_cfg = yaml.safe_load(open(BASE_DIR / "configs" / "models.yaml"))
    set_seed(config["project"]["seed"])
    rng = np.random.default_rng(config["project"]["seed"])
    classes = config["project"]["classes"]

    log_dir = BASE_DIR / "log"
    out_dir = ensure_dir(log_dir / "global_results")

    ens_metrics = log_dir / "Ensemble_CNN" / "metrics" / "test_metrics.json"
    if not ens_metrics.exists():
        sys.exit("Ensemble_CNN not found; run --mode ensemble first")
    members = json.load(open(ens_metrics))["ensemble_models"]
    print(f"ensemble members: {' + '.join(members)}")

    print("fitting temperature on validation")
    val_logits, y_val = validation_logits(members, config, models_cfg)
    T = {m: fit_temperature(val_logits[m], y_val) for m in members}
    for m in members:
        print(f"  [{m}] T = {T[m]:.3f}")

    p_val = np.mean([softmax(val_logits[m] / T[m]) for m in members], axis=0)

    y_test = np.load(log_dir / "Ensemble_CNN" / "predictions" / "y_true.npy")
    member_prob = {m: np.load(log_dir / m / "predictions" / "y_prob.npy") for m in members}
    p_test_raw = np.mean([member_prob[m] for m in members], axis=0)
    p_test = np.mean([softmax(np.log(np.clip(member_prob[m], 1e-12, None)) / T[m])
                      for m in members], axis=0)

    calib = {
        "validation_calibrated": calibration_report(p_val, y_val),
        "test_uncalibrated": calibration_report(p_test_raw, y_test),
        "test_calibrated": calibration_report(p_test, y_test),
    }
    print("\ncalibration")
    for k, v in calib.items():
        print(f"  {k:24s} acc {v['accuracy']:.4f}  ECE {v['ece']:.4f}  "
              f"Brier {v['brier']:.4f}  NLL {v['nll']:.4f}")

    conf_val, corr_val = p_val.max(1), (p_val.argmax(1) == y_val)
    conf_test, corr_test = p_test.max(1), (p_test.argmax(1) == y_test)

    print("\npolicy (threshold fitted on validation, applied once to test)")
    rows = []
    for target in args.targets:
        order = np.argsort(-conf_val)
        risk_cum = np.cumsum(~corr_val[order]) / np.arange(1, len(order) + 1)
        ok = np.where(risk_cum <= target)[0]
        tau = float(conf_val[order][int(ok.max()) if len(ok) else 0])

        accepted = conf_test >= tau
        n_acc, n_err = int(accepted.sum()), int((~corr_test[accepted]).sum())
        risk = n_err / n_acc if n_acc else float("nan")

        boot = []
        idx = np.arange(len(y_test))
        for _ in range(N_BOOTSTRAP):
            s = rng.choice(idx, len(idx), replace=True)
            a = conf_test[s] >= tau
            if a.sum():
                boot.append((~corr_test[s][a]).mean())
        lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (float("nan"),) * 2
        # With zero observed errors the bootstrap interval collapses to [0, 0];
        # the one-sided Clopper-Pearson limit is the honest upper bound there.
        upper = 1.0 if n_acc == 0 or n_err >= n_acc else float(beta.ppf(0.95, n_err + 1, n_acc - n_err))

        rows.append({"target": target, "threshold": tau,
                     "val_coverage": float((conf_val >= tau).mean()),
                     "val_risk": float((~corr_val[conf_val >= tau]).mean()),
                     "test_coverage": float(accepted.mean()), "test_risk": risk,
                     "test_n_accepted": n_acc, "test_errors": n_err,
                     "test_risk_ci": [lo, hi], "test_risk_upper95": upper})
        print(f"  target {target:>5.0%}  tau {tau:.4f} | test coverage {accepted.mean():.3f}  "
              f"risk {risk:.4f}  CI [{lo:.4f}, {hi:.4f}]  upper95 {upper:.4f}")

    # Per-class behaviour at the first target above zero: a single global
    # threshold does not spread risk evenly across classes.
    ref = next((r for r in rows if r["target"] > 0), rows[0])
    accepted = conf_test >= ref["threshold"]
    per_class = {}
    print(f"\nper class at the {ref['target']:.0%} target (tau = {ref['threshold']:.4f})")
    for i, c in enumerate(classes):
        m = y_test == i
        acc_m = accepted & m
        risk = float((~corr_test[acc_m]).mean()) if acc_m.sum() else float("nan")
        per_class[c] = {"coverage": float(accepted[m].mean()), "risk": risk,
                        "errors": int((~corr_test[acc_m]).sum())}
        print(f"  {c:20s} coverage {per_class[c]['coverage']:.3f}  risk {risk:.4f}")

    out = out_dir / "abstention_policy.json"
    json.dump({"members": members, "temperature": T, "calibration": calib,
               "policy": rows, "per_class": per_class}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")

    from plots import plot_risk_coverage_curve
    fig = plot_risk_coverage_curve(conf_test, corr_test, rows, out_dir)
    print(f"wrote {fig}")


if __name__ == "__main__":
    main()
