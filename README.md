# Colorectal Cancer Tissue Classifier

This project classifies colorectal-cancer histology image patches into the nine PathMNIST tissue classes. It contains a small PyTorch training and evaluation pipeline, three model variants, exploratory analysis, and an explainability notebook.

## Task and data

- Dataset: [PathMNIST](https://medmnist.com/), loaded through the `medmnist` package.
- Input: RGB image patches resized to `28 × 28` and normalized channel-wise.
- Target: nine tissue classes: ADI (adipose), BACK (background), DEB (debris), LYM (lymphocytes), MUC (mucus), MUS (smooth muscle), NORM (normal colon mucosa), STR (cancer-associated stroma), and TUM (colorectal adenocarcinoma epithelium).
- Splits: the MedMNIST train, validation, and test splits.

The data loader calls MedMNIST with `download=True`, so the dataset is downloaded into `datasets/` when a loader is first created. The separate `scripts/setup_data.py` Kaggle bootstrap is not required by the training pipeline.

## Implemented models

The model factory in `src/models.py` exposes:

- `kmeans`: a trainable centroid-distance baseline. It flattens each image and returns negative squared distances to nine learnable centroids; it is not an external clustering package.
- `cnn_baseline`: a shallow CNN with two convolution, batch-normalization, ReLU, and max-pooling blocks followed by a small fully connected classifier.
- `resnet18`: torchvision ResNet-18 adapted for small images by replacing the initial `7 × 7`/stride-2 convolution with a `3 × 3`/stride-1 convolution and removing the initial max-pool.

All models are trained with cross-entropy loss and Adam. The best checkpoint is selected by validation accuracy.

## Pipeline

```text
MedMNIST loader
      │
      ├── train split ──► model training ──► best_model.pth
      ├── validation ───► checkpoint selection
      └── test split ───► accuracy + macro one-vs-rest AUC

single image ──► resize/normalize ──► selected model ──► class + confidence
```

The notebooks provide additional EDA and XAI work. `notebooks/eda.ipynb` includes dataset inspection and model analysis; `notebooks/xai.ipynb` contains saliency-map, Grad-CAM, and LIME experiments for ResNet-18.

## Repository structure

```text
config/config.yaml       Default data and training settings
src/dataset.py           MedMNIST datasets and PyTorch DataLoaders
src/models.py            Model implementations and factory
scripts/setup_data.py    Separate Kaggle download helper
scripts/train.py         Training CLI and run-directory creation
scripts/evaluate.py      Test-set evaluation CLI
scripts/predict.py       Single-image inference CLI
scripts/view_image.py    Image preview helper
notebooks/eda.ipynb      Exploratory analysis
notebooks/xai.ipynb      Explainability experiments
runs/                    Generated checkpoints and metrics
requirements.txt         Python dependencies
```

## Setup and usage

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train a model from the repository root. The default model is `kmeans`; select another supported model explicitly when needed:

```bash
python scripts/train.py --model resnet18 --epochs 15 --batch_size 64
```

Each run writes a timestamped directory such as `runs/resnet18/run_YYYYMMDD_HHMMSS/`, containing the best checkpoint and a saved configuration snapshot. Evaluation can use that run directory:

```bash
python scripts/evaluate.py \
  --run_path runs/resnet18/run_YYYYMMDD_HHMMSS
```

For a single image, provide a compatible checkpoint and an image path:

```bash
python scripts/predict.py \
  --image_path path/to/image.png \
  --model resnet18 \
  --weights_path runs/resnet18/run_YYYYMMDD_HHMMSS/best_model.pth
```

## Results and pretrained weights

The [v1.0-models GitHub release](https://github.com/WiktorZoga/colorectal-cancer-classifier/releases/tag/v1.0-models) reports results for pretrained models trained on NCT-CRC-HE-100K and evaluated on CRC-VAL-HE-7K, using nine tissue classes:

| Model | Test accuracy | Test AUC |
| --- | ---: | ---: |
| Adapted ResNet-18 | 90.14% | 0.9875 |
| Shallow CNN | 83.68% | 0.9644 |

The release includes pretrained PyTorch weights. The v1.0 release reports results associated with the original NCT-CRC-HE-100K / CRC-VAL-HE-7K workflow. PathMNIST is derived from this data lineage; the current repository loader in `src/dataset.py` uses MedMNIST’s standardized 28 × 28 PathMNIST representation and its train, validation, and test splits. The release and current-code metrics should only be directly compared when their preprocessing and evaluation details are known to match.

### Historical repository result

The repository history also contains a recorded `kmeans` run with:

| Model | Test accuracy | Macro one-vs-rest AUC |
| --- | ---: | ---: |
| `kmeans` centroid baseline | 31.73% | 0.7014 |

These are historical recorded results from the repository.

## Limitations and future work

The release metrics and current MedMNIST workflow describe different dataset stages/workflows, so they should not be compared as a single controlled experiment without confirming the data lineage. Hyperparameter sweeps, broader model comparison, and a formal comparison with external benchmarks remain future work.
