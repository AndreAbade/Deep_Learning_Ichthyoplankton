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

**Ensemble (1)**: mean of the class probabilities of the top-2 CNNs from distinct architecture families, selected automatically by validation balanced accuracy. In the current run this resolved to DenseNet201 + MobileNetV3-Large.

Backbones come from `torchvision`, `timm`, or `ultralytics`, dispatched by a single factory in `src/models.py`.

---

## Results

Held-out test set (n = 1,455), ranked by macro-F1. Running the pipeline regenerates the full table at `log/global_results/ranking_models.csv`, along with every figure and statistical test reported below.

| # | Model | Accuracy | Macro-F1 | Balanced acc. | Macro-AUC |
|---:|---|---:|---:|---:|---:|
| 1 | YOLO26N-CLS | 0.9856 | 0.9856 | 0.9855 | 0.9997 |
| 2 | Ensemble (DenseNet201 + MobileNetV3-Large) | 0.9677 | 0.9675 | 0.9676 | 0.9990 |
| 3 | DenseNet201 | 0.9643 | 0.9641 | 0.9641 | 0.9987 |
| 4 | MobileNetV3-Large | 0.9622 | 0.9620 | 0.9620 | 0.9989 |
| 5 | ResNet152 | 0.9601 | 0.9600 | 0.9600 | 0.9980 |
| … | … | … | … | … | … |
| 20 | VGG19 | 0.9251 | 0.9240 | 0.9247 | 0.9943 |
| 21 | Swin-T | 0.9244 | 0.9232 | 0.9240 | 0.9946 |
| 22 | VGG16 | 0.9223 | 0.9211 | 0.9220 | 0.9939 |

Notes on reading this table:

- **Every model exceeded 92% accuracy.** The field is tightly clustered, which is exactly why the pipeline reports bootstrap confidence intervals and pairwise McNemar tests rather than raw rankings alone.
- **The YOLO result is stated as models-as-configured.** Its augmentation, epoch budget, input resolution, and learning-rate endpoints were matched to the other models, but its optimizer, learning-rate schedule shape, and freeze phase come from the Ultralytics training recipe and differ from the shared two-phase protocol. That residual difference is a real caveat, not a null one.
- **Accuracy tracked efficiency rather than opposing it.** The two smallest models placed first and fourth, ahead of much larger backbones.
- **The ensemble did not beat the best single model**, so its added inference cost is hard to justify here.

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
