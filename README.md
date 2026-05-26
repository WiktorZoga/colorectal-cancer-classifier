# Colorectal Cancer Digital Pathology Classifier

A modular Deep Learning and Machine Learning framework designed to classify 9 distinct tissue types from colorectal cancer histology slides using the **PathMNIST** dataset benchmark.

## Project Structure

```text
colorectal-cancer-classifier/
│
├── config/
│   └── config.yaml          # Global execution parameters
│
├── src/
│   ├── dataset.py           # MedMNIST PyTorch DataLoaders configuration
│   └── models.py            # Model Factory Pattern (K-Means, CNN, ResNet-18)
│
├── scripts/
│   ├── setup_data.py        # Automated local dataset acquisition script
│   ├── train.py             # Flexible training pipeline with CLI overrides
│   ├── evaluate.py          # Symmetrical test evaluation script
│   ├── view_image.py        # Image preview script
│   └── predict.py           # Target single-image inference deployment script
│
├── notebooks/               # Directory for exploratory data analysis
├── runs/                    # Automated local run directory for logging artifacts
├── requirements.txt         # Strict environmental dependencies mapping
└── README.md                # System documentation guide

```

## Environment Setup & Quickstart

### 1. Dependency Installation

Clone the repository, initialize your virtual environment, and install the required libraries:

```bash
pip install -r requirements.txt

```

### 2. API Authorization Setup

Create a private file named `.env` in the root directory of the project. Generate a v2 token on your Kaggle Settings page, copy the string token (`KGAT_...`), and paste it inside:

```env
KAGGLE_API_TOKEN="KGAT_your_secret_token_here"

```

*Note: The `.env` file is hidden locally and explicitly ignored by Git to prevent private credentials leakage.*

### 3. Data Acquisition

Run the data bootstrap script to extract the raw source files and prepare the system storage:

```bash
python scripts/setup_data.py

```

---

## Execution Interface (CLI Usage)

### Training Models

You can run a quick execution loop across any model architecture. You can also pass a custom configuration path file manually to decouple test runs:

```bash
# 1. Execute Classical Baseline (K-Means)
python scripts/train.py --model kmeans --epochs 5

# 2. Execute Custom Convolutional Network
python scripts/train.py --model cnn_baseline --epochs 10 --lr 0.001

# 3. Execute ResNet-18 (Adapted for 28x28 spatial matrices)
python scripts/train.py --model resnet18 --epochs 15 --batch_size 64

# 4. Train using a completely custom configuration file profile
python scripts/train.py --config_path config/my_custom_config.yaml --model cnn_baseline

```

Every training session automatically snapshots the exact `.yaml` configuration state used and stores it inside a time-stamped directory in `runs/<model_name>/run_YYYYMMDD_HHMMSS/`.

### Evaluating Performance

Test performance evaluation can be conducted either sequentially using the explicit tracking path, or completely manual:

```bash
# Symmetrical Run Mode (Recommended)
python scripts/evaluate.py --run_path runs/cnn_baseline/run_xxx

# Explicit Manual Mode Override
python scripts/evaluate.py --model resnet18 --weights_path runs/resnet18/run_xxx/best_model.pth --config_path config/config.yaml

```

### Running Inference (Single Image)

To evaluate the model's confidence on an individual tissue image patch:

```bash
python scripts/predict.py \
  --image_path datasets/imrankhan77/crc-val-he-7k/versions/1/CRC-VAL-HE-7K/xxx/yyy.tif \
  --model cnn_baseline \
  --weights_path runs/cnn_baseline/run_xxx/best_model.pth

```