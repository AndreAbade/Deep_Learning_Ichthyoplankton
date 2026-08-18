# Neotropical Ichthyoplankton Classification Benchmark

A reproducible PyTorch benchmark of 21 deep learning architectures for classifying Neotropical freshwater ichthyoplankton (fish eggs and larvae) in stereomicroscopy images.

Identifying fish eggs and larvae is a foundational task in fisheries science and freshwater biodiversity monitoring, and it remains a manual, expert-dependent bottleneck: early life stages are morphologically similar, and the most reliable confirmation (genetic sequencing) is slow and costly. This repository contains the full pipeline behind a systematic, statistically grounded comparison of modern architectures on this task: data splitting, transfer-learning training, evaluation, bootstrap confidence intervals, pairwise significance testing, and figure generation.

The code supports the manuscript *"Comparative Evaluation of Deep Learning Architectures for Classifying Neotropical Freshwater Ichthyoplankton in Stereomicroscopy Images"* (see [Manuscript](#manuscript)).

---

## Dataset

9,667 stereomicroscopy images across six categories: fish eggs plus five larval/juvenile taxonomic groups. The split is stratified at 70/15/15 with a fixed seed (42).

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Anostomidae | 1,121 | 240 | 241 | 1,602 |
| Characiformes | 1,124 | 241 | 242 | 1,607 |
| Eggs | 1,148 | 246 | 246 | 1,640 |
| Pimelodidae | 1,125 | 241 | 242 | 1,608 |
| Prochilodontidae | 1,124 | 240 | 242 | 1,606 |
| Siluriformes | 1,122 | 240 | 242 | 1,604 |
| **Total** | **6,764** | **1,448** | **1,455** | **9,667** |

The classes are near-balanced by design, so accuracy and balanced accuracy stay close throughout.

Image data is not distributed with this repository. The pipeline expects the originals under `DataSet/`, one folder per class:

```
DataSet/
├── Anostomidae/
├── Characiformes/
├── Eggs/
├── Pimelodidae/
├── Prochilodontidae/
└── Siluriformes/
```

`python src/main.py --mode split_data` reads that tree and builds `data/{train,val,test}` as symlinks, so the originals are never modified or duplicated.

---

## Models benchmarked

21 architectures plus a probability-averaging ensemble.

**Convolutional networks (18)**: DenseNet121, DenseNet201, VGG16, VGG19, ResNet50, ResNet101, ResNet152, ResNeXt50, InceptionV3, Xception, InceptionResNetV2, NASNetLarge, MobileNetV2, MobileNetV3-Large, EfficientNetV2-B0, EfficientNetV2-M, NFNet (eca_nfnet_l0), ConvNeXt-Tiny

**Vision Transformers (2)**: ViT-B/16, Swin-T

**YOLO (1)**: YOLO26N-CLS

**Ensemble (1)**: mean of the class probabilities of the top-2 CNNs from distinct architecture families, selected automatically by validation balanced accuracy. In the current run this resolves to **DenseNet121 + MobileNetV3-Large**. Selection uses validation performance only, never the test set.

Backbones come from `torchvision`, `timm`, or `ultralytics`, dispatched by a single factory in `src/models.py`.

---

## Results

### Primary result — all models under the shared protocol

Held-out test set (n = 1,455), ranked by macro-F1. Every model below is trained under the shared
augmentation pipeline (`configs/augmentations.yaml`) and the same epoch budget, input resolution and
learning-rate endpoints, so the numbers are mutually comparable. Running the pipeline regenerates the
full table at `log/global_results/ranking_models.csv`, along with every figure and statistical test.

| # | Model | Accuracy | Macro-F1 | Balanced acc. | Macro-AUC |
|---:|---|---:|---:|---:|---:|
| 1 | **Ensemble (DenseNet121 + MobileNetV3-Large)** | **0.9684** | **0.9682** | **0.9682** | **0.9990** |
| 2 | DenseNet201 | 0.9643 | 0.9641 | 0.9641 | 0.9987 |
| 3 | MobileNetV3-Large | 0.9622 | 0.9620 | 0.9620 | 0.9989 |
| 4 | ResNet152 | 0.9601 | 0.9600 | 0.9600 | 0.9980 |
| 5 | ResNet50 | 0.9588 | 0.9586 | 0.9586 | 0.9979 |
| 6 | NFNet (eca_nfnet_l0) | 0.9588 | 0.9584 | 0.9586 | 0.9986 |
| 7 | DenseNet121 | 0.9567 | 0.9565 | 0.9565 | 0.9983 |
| … | … | … | … | … | … |
| 18 | InceptionV3 | 0.9381 | 0.9375 | 0.9379 | 0.9957 |
| 19 | **YOLO26N-CLS** (augmentation harmonised) | **0.9340** | **0.9338** | **0.9338** | **0.9952** |
| 20 | VGG19 | 0.9251 | 0.9240 | 0.9247 | 0.9943 |
| 21 | Swin-T | 0.9244 | 0.9232 | 0.9240 | 0.9946 |
| 22 | VGG16 | 0.9223 | 0.9211 | 0.9220 | 0.9939 |

Notes on reading this table:

- **Every model exceeded 92% accuracy.** The field is tightly clustered, which is exactly why the pipeline reports bootstrap confidence intervals and pairwise McNemar tests rather than raw rankings alone.
- **The leader is not statistically separated from the models behind it.** The ensemble does not differ significantly from DenseNet201 (p = 0.42), MobileNetV3-Large (p = 0.14), ResNet152 (p = 0.16) or NFNet (p = 0.07). Read the upper table as a tie, not a ranking.
- **Size and accuracy are not significantly associated** (Spearman ρ = −0.26, p = 0.26). MobileNetV3-Large at 5.5 M parameters ranking third, ahead of backbones 10–16× larger, is the clearest efficiency result here — but the smallest architecture in the field ranks nineteenth, so "smaller is better" is not supported.
- **The two-tier structure that is firmly established** is the upper group against the trailing group (VGG16, Swin-T, VGG19). After FDR correction, 106 of 231 pairwise McNemar tests remain significant (116 uncorrected).

### Secondary analysis — sensitivity to the training recipe

`YOLO26N_CLS` is the one model whose native framework (Ultralytics) exposes a different and narrower
augmentation API than the shared `torchvision` pipeline. We report both configurations, and the
contrast is a result in its own right:

| Configuration | Accuracy | Macro-F1 | Rank | Status |
|---|---:|---:|---:|---|
| Augmentation harmonised to the shared pipeline | 0.9340 | 0.9338 | 19 of 22 | **primary** — the comparable number |
| Ultralytics defaults (crop 0.5–1.0, RandAugment, random erasing, HSV jitter) | 0.9856 | 0.9856 | 1 of 22 | **sensitivity analysis only** — not comparable |

Epoch budget, input resolution and learning-rate endpoints are identical in both runs, so the
5.2-percentage-point gap is attributable to the augmentation regime rather than to the architecture.

Two caveats on the harmonised run, stated precisely:

- Harmonisation is **partial**. Crop range, flips, RandAugment, erasing and colour jitter were matched; two shared operations, random rotation (20°) and affine translation (0.15), cannot be expressed through the Ultralytics classification API and remain unmatched in the opposite direction.
- The framework's default optimizer, learning-rate schedule shape and absence of an explicit freeze phase are still unmatched. Its nineteenth place is therefore a property of the model as configured, exactly as its earlier first place was.

An earlier version of this repository reported the 0.9856 run as the headline result. It is retained
above as a sensitivity analysis because the comparison is informative: statistical testing did not
detect this confound — the earlier result was significant after FDR correction and still misleading
as an architectural claim — whereas matching the training recipe did.

### Statistical analysis

- **Bootstrap**: 1,000 resamples of the test set per model, producing 95% CIs for six metrics (`bootstrap_results.csv`, violin plots, CI bars, heatmaps).
- **McNemar**: all pairwise comparisons on paired test-set predictions, with false-discovery-rate correction (`mcnemar_results.csv`, heatmap of −log₁₀(p), and a network graph where node size scales with F1 and edge colour encodes significance).

---

## Installation

Requires Python 3.10 and a CUDA-capable GPU (the reference runs used 2× RTX A5500, 24 GB each, CUDA 12.7, PyTorch 2.1.2). Multi-GPU is picked up automatically via `nn.DataParallel`.

```bash
python -m venv env_torch
source env_torch/bin/activate
pip install -r requirements.txt
```

The `RAM cache` loader holds the decoded image set in memory to keep the GPUs fed; the full dataset needs roughly 18 GB of free RAM. Reduce `training.num_workers` in `configs/config.yaml` if you hit DataLoader worker crashes.

---

## Usage

A single CLI entry point drives every stage:

```bash
# End-to-end: split, train all models, evaluate, ensemble, bootstrap, McNemar, report
python src/main.py --mode full_pipeline
```

Individual stages:

```bash
python src/main.py --mode split_data              # build data/ symlink tree
python src/main.py --mode train_all               # skips models with an existing checkpoint
python src/main.py --mode train_all --force       # retrain from scratch
python src/main.py --mode train_one --model ResNet50
python src/main.py --mode evaluate_all            # inference + bootstrap + McNemar
python src/main.py --mode ensemble
python src/main.py --mode bootstrap
python src/main.py --mode mcnemar
python src/main.py --mode report                  # log/global_results/report.md
```

### Configuration

| File | Contents |
|---|---|
| `configs/config.yaml` | Seed, paths, split ratios, training hyperparameters, bootstrap settings |
| `configs/models.yaml` | Model list with backend, backbone key, batch size, input size, family grouping |
| `configs/augmentations.yaml` | Train-time augmentation and val/test preprocessing |

### Training protocol

All non-YOLO models share one two-phase transfer-learning protocol:

| Parameter | Value |
|---|---|
| Optimizer | SGD (momentum 0.9, weight decay 1e-4) |
| Scheduler | CyclicLR, `triangular2`, stepped per batch |
| Learning rate | 0.0001 → 0.006, `step_size_up` = 5 epochs |
| Phase 1 | 5 epochs, backbone frozen, LR fixed at `max_lr` |
| Phase 2 | 50 epochs total, backbone unfrozen, LR cycling |
| Checkpoint selection | Best validation balanced accuracy |

---

## Outputs

Per model, under `log/<MODEL_NAME>/`:

```
checkpoints/     best_model.pt, last_model.pt
predictions/     y_true.npy, y_pred.npy, y_prob.npy, filenames.csv
metrics/         history.csv, best_epoch_metrics.json, test_metrics.json,
                 classification_report.csv, hyperparams.json
figures/         confusion matrices (raw + normalized), multiclass ROC,
                 loss/accuracy curves, metric radar
config_used.yaml, environment.txt
```

Cross-model artifacts, under `log/global_results/`:

- `ranking_models.csv`, `global_summary.json`, `hyperparams_table.csv`, `overfitting_analysis.csv`
- `bootstrap_results.csv`, `mcnemar_results.csv`
- Comparison figures: multi-metric radar of the top 5, grouped bar chart, parallel coordinates, learning-curve grid, fit scatter and heatmap
- Statistical figures: bootstrap violins, CI bars and heatmaps per metric; McNemar heatmap; model network graph
- `report.md`, an auto-generated summary of the whole run

Every model writes `config_used.yaml` and `environment.txt` next to its checkpoints, so any single result can be traced back to the exact configuration and package versions that produced it.

---

## Repository layout

```
├── configs/            # config.yaml, models.yaml, augmentations.yaml
├── src/
│   ├── main.py         # CLI entry point, dispatches all modes
│   ├── dataset.py      # stratified split, ImageFolder loaders, RAM cache
│   ├── models.py       # factory: torchvision / timm / ultralytics
│   ├── train.py        # two-phase training loop
│   ├── evaluate.py     # test-set inference, per-model metrics
│   ├── metrics.py      # 7 metrics, incl. macro specificity and macro AUC
│   ├── plots.py        # per-model and cross-model figures
│   ├── ensemble.py     # top-2 CNN probability averaging
│   ├── bootstrap.py    # 1,000-iteration resampling
│   ├── mcnemar.py      # pairwise significance testing
│   ├── overfitting.py  # hyperparameter table and fit analysis
│   ├── yolo_runner.py  # Ultralytics wrapper
│   ├── report.py       # report.md generation
│   └── utils.py        # seeding, device selection, config capture
├── DataSet/            # original images, supplied by the user; read-only to the pipeline
├── data/               # split symlinks, generated
└── log/                # training and evaluation artifacts, generated
```

`DataSet/`, `data/`, and `log/` are not part of this repository: the first is the input you provide, the other two are produced by running the pipeline.

---

## Reproducibility

- Global seed 42 across NumPy, PyTorch, and CUDA; `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set at module import in `src/utils.py`, before any CUDA call.
- The train/val/test split is stratified and seed-derived, so it reconstructs identically from `DataSet/`.
- Bootstrap resampling uses its own fixed `random_state`.
- Metrics: accuracy, macro-F1, macro precision, macro recall, macro specificity, balanced accuracy, macro AUC. Macro specificity is computed from the confusion matrix as the per-class mean of TN/(TN+FP); macro AUC uses one-vs-rest averaging over predicted probabilities.

GPU nondeterminism in some cuDNN kernels means metrics can differ in the last decimal places between runs on different hardware.

---

## Manuscript

This repository is the code availability companion to the manuscript *"Comparative Evaluation of Deep Learning Architectures for Classifying Neotropical Freshwater Ichthyoplankton in Stereomicroscopy Images"*, submitted to *Ecological Informatics* (Elsevier). It contains the benchmark pipeline: training, evaluation, bootstrap, and McNemar analysis.

The manuscript text, its figures, and the image dataset are not distributed here. For access to the imaged material, contact the corresponding author.

### Which version of this code produced the reported results

The results in the manuscript, and in [Results](#results) above, were produced by the commit tagged
**`v2.0-matched-augmentation`**. Use that tag rather than the branch tip when reproducing the paper:

```bash
git clone https://github.com/AndreAbade/Deep_Learning_Ichthyoplankton.git
cd Deep_Learning_Ichthyoplankton
git checkout v2.0-matched-augmentation
```

| Tag | What it corresponds to |
|---|---|
| `v2.0-matched-augmentation` | **Current.** All 21 architectures under the shared augmentation pipeline. Leading model: the DenseNet121 + MobileNetV3-Large ensemble at 0.9684. `YOLO26N_CLS` is nineteenth at 0.9340. |
| `v1.0-ultralytics-defaults` | Superseded. `YOLO26N_CLS` trained on the Ultralytics default augmentation and reported as the headline result at 0.9856. Retained so the sensitivity analysis in [Results](#results) can be reproduced; **not** the configuration the manuscript's primary results come from. |

Only `src/yolo_runner.py` (the augmentation settings) and `src/scale_robustness.py` (which model the
probe follows) differ between the two tags. No other model was retrained, and every other model's
metrics are identical across both.

---

## Citation

If you use this code or the benchmark protocol, please cite:

```bibtex
@article{abade_ichthyoplankton_benchmark,
  author  = {Abade, Andr\'{e} da Silva and Carnicer, Cleide and
             Oliveira, Bet\^{a}nia Arcanjo de and Lima Junior, Dilermando Pereira},
  title   = {Comparative Evaluation of Deep Learning Architectures for Classifying
             Neotropical Freshwater Ichthyoplankton in Stereomicroscopy Images},
  note    = {Manuscript under review}
}
```

## License

Code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)). The licence covers the source code and configuration only. The image dataset and the manuscript text are not distributed here and are not covered by it.

## Authors

- **André da Silva Abade** (corresponding author), Division of Computer Vision, Federal Institute of Education, Science and Technology of Mato Grosso, Barra do Garças, MT, Brazil (<andre.abade@ifmt.edu.br>)
- **Cleide Carnicer**, **Betânia Arcanjo de Oliveira**, **Dilermando Pereira Lima Junior**, Laboratório de Ecologia e Conservação de Ecossistemas Aquáticos (LECEA), Universidade Federal de Mato Grosso, Pontal do Araguaia, MT, Brazil

The imaged material was collected in the River Araguaia, in the Tocantins–Araguaia basin.
