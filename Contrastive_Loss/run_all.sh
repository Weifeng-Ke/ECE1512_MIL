#!/bin/bash

# Directory containing config files
CONFIG_DIR="config_auto"

# Output directory for logs
OUTPUT_DIR="logs"
mkdir -p "$OUTPUT_DIR"

# Loop over all .yml files in the config directory
for CONFIG_FILE in "$CONFIG_DIR"/*.yml; do
    # Get the base name of the config file (e.g., bracs1.yml -> bracs1)
    BASE_NAME=$(basename "$CONFIG_FILE" .yml)
    
    # Run Python script and save stdout & stderr to a log file
    python batched_main.py --config "$CONFIG_FILE" > "$OUTPUT_DIR/${BASE_NAME}.log" 2>&1
    
    echo "Finished $CONFIG_FILE, output saved to $OUTPUT_DIR/${BASE_NAME}.log"
done
