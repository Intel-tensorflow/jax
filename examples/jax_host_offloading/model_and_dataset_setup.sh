#!/bin/bash
# filepath: model_and_dataset_setup.sh

# Default values
VARIANT="28b"
IMG_SIZE="224"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --variant=*)
            VARIANT="${1#*=}"
            ;;
        --img-size=*)
            IMG_SIZE="${1#*=}"
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

echo "Setting up PaliGemma-2 model ($VARIANT) with image size ${IMG_SIZE}px²..."

# Set model configuration based on variant
case "$VARIANT" in
    "3b")
        LLM_VARIANT="gemma2_2b"
        MODEL_PATH="./paligemma2-3b-pt-${IMG_SIZE}.b16.npz"
        KAGGLE_HANDLE="google/paligemma-2/jax/paligemma2-3b-pt-${IMG_SIZE}"
        ;;
    "10b")
        LLM_VARIANT="gemma2_9b"
        MODEL_PATH="./paligemma2-10b-pt-${IMG_SIZE}.b16.npz"
        KAGGLE_HANDLE="google/paligemma-2/jax/paligemma2-10b-pt-${IMG_SIZE}"
        ;;
    "28b")
        LLM_VARIANT="gemma2_27b"
        MODEL_PATH="./paligemma2-28b-pt-${IMG_SIZE}.b16.npz"
        KAGGLE_HANDLE="google/paligemma-2/jax/paligemma2-28b-pt-${IMG_SIZE}"
        ;;
  *)
    echo "Error: Unsupported variant '$VARIANT'. Use 3b, 10b, or 28b."
    exit 1
    ;;
esac

echo "Using model configuration:"
echo "- LLM variant: $LLM_VARIANT"
echo "- Model path: $MODEL_PATH"
echo "- Kaggle handle: $KAGGLE_HANDLE"

# Download model checkpoint if it doesn't exist
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Downloading the checkpoint from Kaggle, this could take a few minutes...."
  
  # Check if kagglehub is installed
  if ! pip show kagglehub &>/dev/null; then
    echo "Installing kagglehub..."
    pip install kagglehub
  fi
  
  # Download the model
  python -c "import kagglehub; kagglehub.model_download('$KAGGLE_HANDLE', '$MODEL_PATH')"
  echo "Model path: $MODEL_PATH"
else
  echo "Model checkpoint already exists at: $MODEL_PATH"
fi

# Download tokenizer if it doesn't exist
TOKENIZER_PATH="./paligemma_tokenizer.model"
if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "Downloading the model tokenizer..."
  gsutil cp gs://big_vision/paligemma_tokenizer.model $TOKENIZER_PATH
  echo "Tokenizer path: $TOKENIZER_PATH"
else
  echo "Tokenizer already exists at: $TOKENIZER_PATH"
fi

# Download dataset if it doesn't exist
DATA_DIR="./longcap100"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Downloading the dataset..."
  gsutil -m -q cp -n -r gs://longcap100/ .
  echo "Data path: $DATA_DIR"
else
  echo "Dataset already exists at: $DATA_DIR"
fi

echo "Setup complete! All required files are now available."