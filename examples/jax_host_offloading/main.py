import sys
sys.path.append('./big_vision_repo')
sys.path.append('./paligemma2/big_vision_repo')
import os
import datetime
import tensorflow as tf
import ml_collections
from absl import app, flags
from src.train import train

# Don't let TF use the GPU or TPUs
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

KAGGLE_CACHE_DIR = os.getenv("KAGGLE_CACHE_DIR", os.path.expanduser("~/.cache/kagglehub"))

FLAGS = flags.FLAGS

# Define individual config flags with defaults
# LLM config
flags.DEFINE_string('llm_variant', 'gemma2_2b', 'LLM variant')
flags.DEFINE_integer('vocab_size', 257_152, 'Vocabulary size')
flags.DEFINE_float('final_logits_softcap', 0.0, 'Final logits softcap value')

# Image model config
flags.DEFINE_string('img_variant', 'So400m/14', 'Image model variant')
flags.DEFINE_string('pool_type', 'none', 'Pooling type')
flags.DEFINE_boolean('scan', True, 'Whether to use scanning')
flags.DEFINE_string('dtype_mm', 'float16', 'Data type for model')

# Training config
flags.DEFINE_string('log_file', 'train.logs', 'Log file path')
flags.DEFINE_string('logdir', './logs', 'Log directory path')
flags.DEFINE_string('tokenizer_path', './paligemma_tokenizer.model', 'Path to tokenizer model')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training')
flags.DEFINE_integer('train_examples', 512, 'Number of training examples')
flags.DEFINE_float('learning_rate', 0.03, 'Learning rate')
flags.DEFINE_string('data_dir', './longcap100', 'Data directory')
flags.DEFINE_integer('seq_length', 128, 'Sequence length')
flags.DEFINE_integer('warmup_iter', 5, 'warm up iterations')
flags.DEFINE_string('device', 'tpu', 'Device to use (tpu or gpu)')
flags.DEFINE_string('offload_policy', 'nothing_saveable', 'A remat policy to apply on Gemma')
flags.DEFINE_bool('compile',True,'if True, the update_fn will be compiled and lowered')
flags.DEFINE_string('optim','sgd', 'Optimizer selection. only \"sgd\" is supported')
flags.DEFINE_string('logs_extra_suffix',None,'Extra suffix for log folder')
# Add profiling flag
flags.DEFINE_boolean('enable_profiler', False, 'Start the JAX profiler server')
flags.DEFINE_integer('profiler_port', 6006, 'Port for the JAX profiler server')


def create_log_dir():
    # Create a timestamped directory based on configuration
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = FLAGS.llm_variant.replace('/', '_').replace('-', '_')
    compile_str = "compiled" if FLAGS.compile else "jitted"
    remat_policy = FLAGS.offload_policy.replace('/', '_').replace('-', '_')
    
    extra_suffix = FLAGS.logs_extra_suffix
    optim = FLAGS.optim
    # Create the log directory name with all the relevant information
    log_dir_name = f"{model_name}_{compile_str}_{remat_policy}_{optim}_{current_time}"
    if extra_suffix is not None:
        log_dir_name = f"{log_dir_name}_{extra_suffix}"
    base_log_dir = FLAGS.logdir

    # Create full path and make sure the directory exists
    session_logdir = os.path.join(base_log_dir, log_dir_name)
    os.makedirs(session_logdir, exist_ok=True)
    return session_logdir



def main(argv):
    # Skip program name
    del argv

    # Create train_config from flags
    train_config = ml_collections.ConfigDict()
    train_config.log_file = FLAGS.log_file
    train_config.logdir = create_log_dir()
    train_config.tokenizer_path = FLAGS.tokenizer_path
    train_config.kaggle_cache_dir = KAGGLE_CACHE_DIR
    train_config.batch_size = FLAGS.batch_size
    train_config.train_examples = FLAGS.train_examples
    train_config.learning_rate = FLAGS.learning_rate
    train_config.data_dir = FLAGS.data_dir
    train_config.seq_length = FLAGS.seq_length
    train_config.device = FLAGS.device
    train_config.compile=FLAGS.compile
    train_config.warmup_iter = FLAGS.warmup_iter
    train_config.optim = FLAGS.optim
    train_config.offload_policy = FLAGS.offload_policy
    
    print(f'Offload policy is {train_config.offload_policy}')
    offload_policy = train_config.offload_policy

    # Create model_config from flags
    model_config = ml_collections.FrozenConfigDict({
        "llm": {
            "vocab_size": FLAGS.vocab_size, 
            "variant": FLAGS.llm_variant, 
            "final_logits_softcap": FLAGS.final_logits_softcap,
            "remat_policy":offload_policy
        },
        "img": {
            "variant": FLAGS.img_variant, 
            "pool_type": FLAGS.pool_type, 
            "scan": FLAGS.scan, 
            "dtype_mm": FLAGS.dtype_mm
        }
    })

    print('=== Model Configuration ===')
    print(model_config)

    print('=== Training Configuration ===')
    print(train_config)
    print(f"Logs will be saved to: {train_config.logdir}")
   
    train(model_config, train_config)


if __name__ == '__main__':
    app.run(main)