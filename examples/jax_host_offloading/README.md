# JAX Host Offloading Example

This example is adapted from https://ai.google.dev/gemma/docs/paligemma/fine-tuning-paligemma.

## PaliGemma2

Follow instructions below for enviroment and model setup.

### Environment Setup

#### 1. Create Conda Environmet

If `Miniforge` not installed, install it.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-$(uname)-$(uname -m).sh
```
Create conda environment.

```bash
conda create -n paligemma2 python=3.12 
```

Once environment is created, activate the environment.

```bash
conda activate paligemma2
```

#### 2. Environment Setup

Run `env_setup_tpu.sh` (TPU-only support):

```bash
./env_setup_tpu.sh --platform=tpu
```

#### 3. Model Setup

1. **Get access to PaliGemma**

    Before using PaliGemma for the first time, you must request access to the model through Kaggle by completing the following steps:

    1. Log in to [Kaggle](https://www.kaggle.com), or create a new Kaggle account if you don't already have one.
    1. Go to the [PaliGemma model card](https://www.kaggle.com/models/google/paligemma-2) and click **Request Access**.
    1. Complete the consent form and accept the terms and conditions.

2. **Configure your API key**

    To use PaliGemma, you must provide your Kaggle username and a Kaggle API key.

    To generate a Kaggle API key, open your [**Settings** page in Kaggle](https://www.kaggle.com/settings) and click **Create New Token**. This triggers the download of a `kaggle.json` file containing your API credentials.

    Then, copy `kaggle.json` to `~/.kaggle`

    ```bash
    cp kaggle.json ~/.kaggle/
    ```

3. **Download Models and Dataset**

    ```bash
    ./model_and_dataset_setup.sh --variant=28b
    ```

    *Note: For 28b model you need more than `50GB` disk space*

#### 4. Activation Offloading

1. Apply the big_vision patch

```bash
cd big_vision_repo
git apply ../big_vision_0701.patch
cd ..
```

2. Train without activation offloading

```bash
python main.py --llm_variant=gemma2_27b --compile=true --logdir=./logs --learning_rate=0.00075 --seq_length=256 --batch_size=128 --device=tpu --optim=sgd --train_examples=8192 --offload_policy=full_remat
```

3. Train with activation offloading

```bash
python main.py --llm_variant=gemma2_27b --compile=true --logdir=./logs --learning_rate=0.00075 --seq_length=256 --batch_size=128 --device=tpu --optim=sgd --train_examples=8192 --offload_policy=qkv_proj_offloaded
```

Training logs will be saved in `./logs/`