# GNSS Jamming Signal Classification

Implementation code for the paper:

> **Machine Learning Image Recognition for GNSS Jamming Signals Categorization**
> J. Steiner, J. Pešík
> *Neural Network World* 6/2024, pp. 341–360
> DOI: [10.14311/NNW.2024.34.019](https://doi.org/10.14311/NNW.2024.34.019)

## Overview

This project uses ResNet-based image recognition to automatically categorize GNSS jamming signals. The input images are spectrum snapshots and waterfall diagrams produced by the GSS100D GPS L1 jamming detector. Five ResNet variants (18, 34, 50, 101, 152 layers) are trained and evaluated via 4-fold cross-validation, with the best models achieving over 90% classification accuracy.

## Jamming Signal Categories

Seven categories from the STRIKE3 project taxonomy are used:

| ID | Category | Description |
|----|----------|-------------|
| ST | Single tone | Single dominant tone; single near-vertical line in waterfall |
| MT | Multi tone | Multiple distinct tones; multiple closely spaced vertical lines in waterfall |
| WSF | Wide sweep fast | Wide frequency sweep with >8 chirps per 100 µs |
| WSS | Wide sweep slow | Wide frequency sweep with 2–7 chirps per 100 µs |
| NS | Narrow sweep | Sweep over a narrow frequency range |
| Sawtooth | Sawtooth | Asymmetric up/down sweep; sharp downward gradient in waterfall |
| Triangular | Triangular | Symmetric up/down sweep; equal-gradient slopes in waterfall |

## Dataset

- **Source:** GSS100D detector installed on Czech highway D1, monitoring campaign May 2021 – March 2022
- **Total events captured:** 2,069 jamming events
- **Dataset used:** 700 images (100 per category), labelled by two human experts
- **Image types:** Spectrum snapshot + waterfall diagram per event
- **Cross-validation:** 4-fold (75 training / 25 validation images per category per fold)

Raw images must be placed in `data/` in subdirectories named `Kategorie <N> - <name>` (matching the regex `Kategorie +\d +- +[\w() ]*`).

## Directory Structure

```
data/                        # Raw labelled images (one subdir per category)
data_processed/              # Preprocessed images split into 4 CV groups
data_active_dir/             # Temporary train/val split for current CV fold
output/                      # Per-class prediction output images
confusion_matrix_output/     # Confusion matrices (.xlsx) per model and fold
Výsledky/                    # Confusion matrix results used for statistical comparison
test_statistics_<N>.csv      # Per-run training statistics for ResNet-<N>
test_statistics_<N>.pdf      # Loss and accuracy plots for ResNet-<N>
```

## Scripts

### `preprocess_images.py`

Shuffles images within each category and distributes them evenly into 4 cross-validation groups under `data_processed/`.

```bash
python preprocess_images.py
```

Run this once before training. It will overwrite `data_processed/`.

### `classify_images.py`

Trains and evaluates a ResNet model using 4-fold cross-validation. Results are saved incrementally to `test_statistics.csv` so training can be resumed if interrupted. Confusion matrices are saved to `confusion_matrix_output/`.

```bash
python classify_images.py <num_epochs> <model> [<model2> ...]
```

**Arguments:**
- `num_epochs` — number of training epochs (paper uses 50)
- `model` — one or more of: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`

**Example:**
```bash
python classify_images.py 50 resnet18 resnet50 resnet152
```

**Hyperparameters (as used in the paper):**
- Optimizer: SGD, lr=0.001, momentum=0.9
- LR scheduler: StepLR, step=7 epochs, gamma=0.1
- Batch size: 5
- Image preprocessing: CenterCrop to 790×340, ImageNet normalization

**Hardware used in the paper:** MacBook Pro, Apple M3 Max (14-core CPU, 30-core GPU, 36 GB shared memory). The code auto-detects Apple MPS; falls back to CPU if unavailable.

### `export_images.py`

Reads all `test_statistics_<N>.csv` files in the current directory and saves loss and accuracy plots as PDFs.

```bash
python export_images.py
```

### `statistical_comparison.py`

Performs a Friedman test to compare classification accuracy across ResNet variants. Reads confusion matrices from the `Výsledky/` directory (files matching `resnet<NN>_<fold>.xlsx`).

```bash
python statistical_comparison.py
```

### `run_statistics.py`

Dataclass definition for per-epoch training statistics. Not run directly.

## Results Summary

| Model | Validation Accuracy | Notes |
|-------|-------------------|-------|
| ResNet-18 | ~90% | Competitive despite fewer layers; fast training |
| ResNet-34 | ~80% | Weakest performer; higher loss values |
| ResNet-50 | ~90% | Stable across all 4 CV subsets |
| ResNet-101 | >85% | Reliable training; high accuracy in second half |
| ResNet-152 | >90% | Best overall; consistently above 75% from early epochs |

Friedman test (α=0.05): statistic=8.468, p=0.076 — no statistically significant difference between variants.

Most common misclassifications: Single tone ↔ Multi tone, Sawtooth ↔ Triangular.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** `torch`, `torchvision`, `pandas`, `matplotlib`, `pillow`, `openpyxl`, `seaborn`, `scipy`

## Citation

```bibtex
@article{steiner2024gnss,
  title   = {Machine Learning Image Recognition for {GNSS} Jamming Signals Categorization},
  author  = {Steiner, J. and Pe{\v{s}}{\'{i}}k, J.},
  journal = {Neural Network World},
  volume  = {34},
  number  = {6},
  pages   = {341--360},
  year    = {2024},
  doi     = {10.14311/NNW.2024.34.019}
}
```
