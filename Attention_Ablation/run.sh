#!/bin/bash

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
#source C:/Users/tanhs/miniconda3/etc/profile.d/conda.sh
#conda activate mil

#python main.py --config config/camelyon16_medical_ssl_config.yml
#python main.py --config config/camelyon17_medical_ssl_config.yml
#python main.py --config config/bracs_medical_ssl_config.yml


#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash run.sh [options]

Options:
  -attn_head, -attn_heads, --attn_head, --attn_heads <N>  Override attn_heads for the run (e.g., 2)
  -camelyon16                               Run only the Camelyon16 config
  -camelyon17                               Run only the Camelyon17 config
  -bracs | -brack                           Run only the BRACS config
  -wandb                                    Enable Weights & Biases logging
  --wandb_project <name>                    Set the Weights & Biases project name (default: ece1512-mil)
  --wandb_entity <entity>                   Optional Weights & Biases entity/team name
  --wandb_run_name <name>                   Optional custom run name for Weights & Biases
  -h | --help                               Show this help message

If no dataset flags are provided, all datasets will be run.
USAGE
}

attn_heads=""
selected_datasets=()
use_wandb= true
wandb_project=""
wandb_entity=""
wandb_run_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -attn_heads | --attn_heads)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for $1" >&2
        usage
        exit 1
      fi
      attn_heads="$2"
      shift 2
      ;;
    -camelyon16)
      selected_datasets+=("config/camelyon16_medical_ssl_config.yml")
      shift
      ;;
    -camelyon17)
      selected_datasets+=("config/camelyon17_medical_ssl_config.yml")
      shift
      ;;
    -bracs)
      selected_datasets+=("config/bracs_medical_ssl_config.yml")
      shift
      ;;
    -wandb)
      use_wandb=true
      shift
      ;;
    --wandb_project)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for $1" >&2
        usage
        exit 1
      fi
      wandb_project="$2"
      shift 2
      ;;
    --wandb_entity)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for $1" >&2
        usage
        exit 1
      fi
      wandb_entity="$2"
      shift 2
      ;;
    --wandb_run_name)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for $1" >&2
        usage
        exit 1
      fi
      wandb_run_name="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$wandb_project" || -n "$wandb_entity" || -n "$wandb_run_name" ]]; then
  use_wandb=true
fi

datasets=(
  "config/camelyon16_medical_ssl_config.yml"
  "config/camelyon17_medical_ssl_config.yml"
  "config/bracs_medical_ssl_config.yml"
)

if [[ ${#selected_datasets[@]} -gt 0 ]]; then
  datasets=("${selected_datasets[@]}")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results_${TIMESTAMP}.txt"

echo "Saving run outputs to ${RESULT_FILE}" | tee -a "$RESULT_FILE"

for config_path in "${datasets[@]}"; do
  echo "\n===== Running ${config_path} =====" | tee -a "$RESULT_FILE"
  python_args=("main.py" "--config" "$config_path")
  if [[ -n "$attn_heads" ]]; then
    python_args+=("--attn_heads" "$attn_heads")
  fi
  if [[ "$use_wandb" == true ]]; then
    python_args+=("--wandb")
  fi
  if [[ -n "$wandb_project" ]]; then
    python_args+=("--wandb_project" "$wandb_project")
  fi
  if [[ -n "$wandb_entity" ]]; then
    python_args+=("--wandb_entity" "$wandb_entity")
  fi
  if [[ -n "$wandb_run_name" ]]; then
    python_args+=("--wandb_run_name" "$wandb_run_name")
  fi

  python "${python_args[@]}" 2>&1 | tee -a "$RESULT_FILE"
  echo "===== Finished ${config_path} =====\n" | tee -a "$RESULT_FILE"
done

echo "All results saved to ${RESULT_FILE}" | tee -a "$RESULT_FILE"
