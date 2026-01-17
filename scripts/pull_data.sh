#!/bin/bash
set -e

# Script to check data integrity and prepare datasets
# If USE_DVC is true, use dvc pull to prepare the data
# Otherwise, use finweb and hellaswag scripts to download datasets

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to log messages with timestamp
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to log errors
error() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Function to log success messages
success() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1"
}

# Function to pull data with DVC
pull_data_with_dvc() {
  log "Pulling data using DVC..."
  # Configure DVC to not require git only if .git directory doesn't exist
  if [ ! -d ".git" ]; then
    log "No .git directory found, configuring DVC to work without git... (It's fine if it's inside a docker container)"
    dvc config core.no_scm true
  fi
  if dvc pull; then
    success "Data successfully pulled with DVC"
    return 0
  else
    error "Failed to pull data with DVC"
    return 1
  fi
}

# Function to download Fineweb dataset
download_fineweb_data() {
  log "Downloading Fineweb dataset..."

  # Use the Python module to download the dataset
  if python -m gpt.fineweb.dataset; then
    success "Fineweb dataset downloaded successfully"
    return 0
  else
    error "Failed to download Fineweb dataset"
    return 1
  fi
}

download_hellaswag_data() {
  log "Downloading HellaSwag..."

  if python -m gpt.hellaswag.dataset; then
    success "HellaSwag downloaded successfully"
    return 0
  else
    error "Failed to download HellaSwag val split"
    return 1
  fi
}

# Main function
main() {
  # Get parameters - no fallback to environment variables
  local use_dvc="$1"

  # Check if all required parameters are provided
  if [ -z "$use_dvc" ]; then
    error "Missing required parameters"
    error "Usage: $0 <USE_DVC>"
    exit 1
  fi

  # Check if USE_DVC is set to true
  if [[ "${use_dvc,,}" =~ ^(1|true|yes|y)$ ]]; then
    log "USE_DVC is set to true, checking if DVC is installed..."

    if command_exists dvc; then
      pull_data_with_dvc || exit 1
    else
      error "DVC is not installed. Please install DVC or set USE_DVC=0"
      exit 1
    fi
  else
    log "USE_DVC is set to false, downloading datasets manually"

    # Download Fineweb dataset
    local fineweb_success=false
    download_fineweb_data && fineweb_success=true

    # Download HellaSwag dataset
    local hellaswag_success=false
    download_hellaswag_data && hellaswag_success=true

    # Check which datasets failed, if any
    local failed_datasets=""

    if [ "$fineweb_success" = false ]; then
      failed_datasets="Fineweb"
    fi

    if [ "$hellaswag_success" = false ]; then
      if [ -n "$failed_datasets" ]; then
        failed_datasets="$failed_datasets, HellaSwag"
      else
        failed_datasets="HellaSwag"
      fi
    fi

    # If there are failed datasets, show error message and exit
    if [ -n "$failed_datasets" ]; then
      error "Failed to download the following datasets: $failed_datasets"
      exit 1
    else
      # All datasets downloaded successfully
      success "All datasets downloaded successfully"
    fi
  fi
}

# Run the main function
# Usage: pull_data.sh <USE_DVC> <FINEWEB_PATH> <CACHE_DIR>
# Example: pull_data.sh 0 ./data/processed/fineweb_edu ./data/.cache
# All parameters are required - script will exit with error if any are missing
main "$@"
