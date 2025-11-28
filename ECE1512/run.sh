#!/bin/bash

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
source C:/Users/tanhs/miniconda3/etc/profile.d/conda.sh
conda activate mil

#python main.py --config config/camelyon16_medical_ssl_config.yml
#python main.py --config config/camelyon17_medical_ssl_config.yml
#python main.py --config config/bracs_medical_ssl_config.yml


set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results_${TIMESTAMP}.txt"

echo "Saving run outputs to ${RESULT_FILE}" | tee -a "$RESULT_FILE"

datasets=(
  "config/camelyon16_medical_ssl_config.yml"
  "config/camelyon17_medical_ssl_config.yml"
  "config/bracs_medical_ssl_config.yml"
)

for config_path in "${datasets[@]}"; do
  echo "\n===== Running ${config_path} =====" | tee -a "$RESULT_FILE"
  python main.py --config "$config_path" 2>&1 | tee -a "$RESULT_FILE"
  echo "===== Finished ${config_path} =====\n" | tee -a "$RESULT_FILE"
done

echo "All results saved to ${RESULT_FILE}" | tee -a "$RESULT_FILE"