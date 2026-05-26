# Project TODO List

## [ ] 1. EDA & Visualizations (Notebooks)
For example: 
* **[ ]** Load dataset in a Notebook using `get_dataloaders()`.
* **[ ]** Display a grid of random images with their actual tissue names (`0 -> ADI`, `3 -> LYM`, etc.).
* **[ ]** Plot a bar chart showing the class distribution (check for dataset imbalance).
* **[ ]** Export charts as `.png` into the `runs/` folder for the final slides.

## [ ] 2. Hyperparameter Tuning 
* **[ ]** Run training loops for all models (`kmeans`, `cnn_baseline`, `resnet18`, maybe some better ones too).
* **[ ]** Experiment with different learning rates, batch sizes, and epochs. Optuna?
* **[ ]** Verify that all training curves (Loss / Accuracy) look good and sync properly on **Weights & Biases (wandb)**.

## [ ] 3. Model Benchmarking & Evaluation
* **[ ]** Run `evaluate.py` for every trained model checkpoint.
* **[ ]** Generate the `evaluation_results.json` files for each run.
* **[ ]** Compare Accuracy and **AUC** metrics between K-Means, CNN, ResNet-18, etc.
* **[ ]** Compare our final test results with the official online **PathMNIST Leaderboard**.

## [ ] 4. Explainable AI (XAI)
* **[ ]** IDK.

## [ ] 5. Project Presentation
* **[ ]** Prepare slides for class explaining the dataset, the baseline setup, results, and XAI findings.

## [ ] 6.  Future Steps
* **[ ]**  ...