#!/bin/bash
# filepath: env_setup_tpu.sh

echo "Setting up environment for PaliGemma multi-device fine-tuning..."

# Parse command line arguments
PLATFORM=""
SKIP_REQUIREMENTS=false
for arg in "$@"; do
  case $arg in
    --platform=*)
      PLATFORM="${arg#*=}"
      ;;
    --skip-requirements)
      SKIP_REQUIREMENTS=true
      ;;
  esac
done

# Clone big_vision repository if it doesn't exist
if [[ ! -d "big_vision_repo" ]]; then
  echo "Cloning big_vision repository..."
  git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo
  echo "Repository cloned successfully."
else
  echo "big_vision repository already exists."
fi

## Install JAX for TPU only
if [[ -z "$PLATFORM" ]]; then
  echo "No platform specified. Use --platform=tpu."
  exit 1
fi

if [[ "$PLATFORM" != "tpu" ]]; then
  echo "Unsupported platform: $PLATFORM. This script only supports --platform=tpu."
  exit 1
fi

echo "Installing JAX with TPU support..."
pip install tensorflow-cpu overrides ml_collections matplotlib sentencepiece --no-cache-dir
pip install "jax[tpu]>=0.5.3" --no-cache-dir
pip install einops flax optax --no-cache-dir

echo "Done installing dependencies!"

echo "Setup complete!"