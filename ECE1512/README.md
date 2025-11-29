# ECE1512 Assignment — MIL Starter Kit

This assignment for ECE1512 asks you to build and evaluate a Multiple Instance Learning (MIL) model on three whole-slide image (WSI) datasets: CAMELYON16, CAMELYON17, and BRACS. We provide pre-extracted patch features in HDF5 (.h5) format, along with a minimal starter codebase.

## What’s Included
* pre-extracted features (https://drive.google.com/drive/folders/1adV3mjdYKLoA2BHhJSHUx9RiCkcR6wIN?usp=drive_link)
  Download the files from Google Drive and place them under the Feature directory (see the layout below).

* Training & evaluation script (main.py)
  An ABMIL baseline you can extend.

* Config-first workflow
  A small YAML config captures dataset paths and dataloader hyperparameters.

## Data Layout

After downloading, organize the feature files like this:

```Feature/
├─ BRACSFeature/
│  └─ patch_feats_pretrain_medical_ssl.h5
├─ camelyon16Feature/
│  └─ patch_feats_pretrain_medical_ssl.h5
└─ camelyon17Feature/
   └─ patch_feats_pretrain_medical_ssl.h5
```

## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Quick start

```bash run.sh```

### Log training to Weights & Biases (wandb)
1. Set your API key once (for example, export `WANDB_API_KEY=<your_key>` or run `wandb login`).
2. Enable logging on any run:
   ```
   bash run.sh -wandb --wandb_project ECE1512_ProjectB --wandb_run_name camelyon16_baseline
   ```
   You can also pass `--wandb_entity <team>` if your project lives under a shared entity.
3. Metrics tracked each epoch: train loss, validation/test AUROC, accuracy, F1, loss, and learning rate. The best-epoch summary is saved to the wandb run as well.

## Acknowledgment

This starter and parts of the training loop are based on and inspired by: ACMIL https://github.com/dazhangyu123/ACMIL

Please credit the original authors if you reuse substantial portions of their code or ideas.