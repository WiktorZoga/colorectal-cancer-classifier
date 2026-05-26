import os
import sys
import yaml
import kagglehub
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ENV_PATH = PROJECT_ROOT / ".env"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

TARGET_DATA_DIR = PROJECT_ROOT / "datasets"
TARGET_DATA_DIR.mkdir(exist_ok=True)

# Explicitly override kagglehub global cache pointer location
os.environ["KAGGLEHUB_CACHE"] = str(TARGET_DATA_DIR)

load_dotenv(dotenv_path=ENV_PATH)
if not os.getenv("KAGGLE_API_TOKEN"):
    raise ValueError(
        "CRITICAL ERROR: Environment variable 'KAGGLE_API_TOKEN' not found inside local .env payload! "
        "Please generate a v2 token on Kaggle settings page and paste it into your local .env file."
    )

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Global configuration profile missing target path context: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

train_dataset = config["data"]["train_dataset"]
test_dataset = config["data"]["test_dataset"]

print("=========================================================================")
print(f"Initializing remote file stream downloads directly to: {TARGET_DATA_DIR}")
print("=========================================================================\n")

print(f"[STREAM 1/2] Fetching target distribution payload: '{train_dataset}'...")
train_path = kagglehub.dataset_download(train_dataset)
print(f"-> Verification Checkpoint Successful. Artifact mapped to: {train_path}\n")

print(f"[STREAM 2/2] Fetching target distribution payload: '{test_dataset}'...")
test_path = kagglehub.dataset_download(test_dataset)
print(f"-> Verification Checkpoint Successful. Artifact mapped to: {test_path}\n")

print("--- PIPELINE INFRASTRUCTURE READINESS ACHIEVED ---")
print("All source files extracted locally. The workspace environment configuration is now operational.")