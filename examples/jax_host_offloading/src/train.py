
import os
import functools
import warnings
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
from jax.experimental.compute_on import compute_on
import sentencepiece


from contextlib import redirect_stdout

# big_vison models
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

# Import big vision utilities
import big_vision.datasets.jsonl
import big_vision.utils
import big_vision.sharding


from src.utils import (parameter_overview, 
                       get_device_memory_usage, 
                       get_data_and_param_shardings,
                       render_examples_to_plot,
                       render_example,
                       render_examples_to_html,
                       plot_examples,
                       fwd_bwd_jaxprs,
                       print_and_save_profiling,
                       print_and_save_compilation_timers)

from src.dataset import (postprocess_tokens, 
                         train_data_iterator, 
                         validation_data_iterator)






def get_model_path(variant, image_size=224, base_dir=None):
    """
    Constructs the path to the PaliGemma2 model based on variant and image size.
    
    Args:
        variant: Model variant (e.g., '3b', '28b')
        image_size: Image size used in pretraining (default: 224)
        base_dir: Base directory for models (default: uses KAGGLE_CACHE_DIR)
    
    Returns:
        String path to the model weights file
    """
    # Use the provided base directory or default to KAGGLE_CACHE_DIR
    if base_dir is None:
       raise ValueError('base_dir which is KAGGLE_CACHE_DIR cannot be None')
    
    # Ensure variant doesn't have 'gemma2_' prefix (common in configs)
    if variant not in ['3b', '10b', '28b']:
        raise ValueError('Invalid PaliGemma2 variant. Valid Values are 3b, 10b or 28b')
    
    # Construct the relative path
    rel_path = f"models/google/paligemma-2/jax/paligemma2-{variant}-pt-{image_size}/1/./paligemma2-{variant}-pt-{image_size}.b16.npz"
    
    # Join with the base directory
    return os.path.join(base_dir, rel_path)

# Define `decode` function to sample outputs from the model.
def get_decode_function(model, tokenizer):
    decode_fn = predict_fns.get_all(model)['decode']
    decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())

def get_model_and_params(model_config, train_config, paligemma_variant, image_size=224):
    print(f'Loading model, tokenizer and parmeters...')
    st = time.time()

    model = paligemma.Model(**model_config)
    tokenizer = sentencepiece.SentencePieceProcessor(train_config.tokenizer_path)
    params = paligemma.load(None, get_model_path(paligemma_variant, image_size, train_config.kaggle_cache_dir), model_config)
    et = time.time() - st
    print(f"Loading model, tokenizer and parmeters completed in {et:.2f} seconds")
    return model, tokenizer, params


def is_trainable_param(name, param):  # pylint: disable=unused-argument
    """Determine if parameter is trainable based on its name."""
    if name.startswith("llm/layers/attn/"):  return True
    if name.startswith("llm/"):              return False
    if name.startswith("img/"):              return False
    raise ValueError(f"Unexpected param name {name}")


def get_sharded_trainable_params(params, params_sharding, log_file=None):
    
    """
    Shard model parameters across available devices.
    
    Args:
        params: Model parameters
        sharding_strategy: Either '1d' for data-only sharding or '2d' for data+model sharding
        log_file: Path to save parameter overview (if None, only prints to console)
    
    Returns:
        Sharded parameters
    """
    # Create trainable parameter mask
    trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

    # Suppress irrelevant warnings
    warnings.filterwarnings(
        "ignore", message="Some donated buffers were not usable")

    # Define a function to cast parameters to appropriate precision
    @functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
    def maybe_cast_to_f32(params, trainable):
        # Cast trainable params to float32, others to float16
        return jax.tree.map(lambda p, m: p.astype(jnp.float32)
                          if m else p.astype(jnp.float16),
                          params, trainable)

    # Flatten parameters for sequential loading to save memory
    params, treedef = jax.tree.flatten(params)
    sharding_leaves = jax.tree.leaves(params_sharding)
    trainable_leaves = jax.tree.leaves(trainable_mask)
    
    print(f"Sharding and casting {len(params)} parameter arrays...")
    start_time = time.time()
    
    # Process each parameter array individually
    for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
        if idx % 10 == 0:
            print(f"Processing parameter {idx}/{len(params)}...")
        params[idx] = big_vision.utils.reshard(params[idx], sharding)
        params[idx] = maybe_cast_to_f32(params[idx], trainable)
        params[idx].block_until_ready()
    
    # Reconstruct parameter tree
    params = jax.tree.unflatten(treedef, params)
    
    elapsed = time.time() - start_time
    print(f"Parameter sharding completed in {elapsed:.2f} seconds")
    
    # Display parameter overview
    print(" == Model params == ")
    log_path = os.path.join("logs", "parameter_overview.txt") if log_file is None else log_file
    parameter_overview(params, log_path)
    
    # Print device memory usage
    get_device_memory_usage('tpu')
    
    return params, trainable_mask

def get_paligemma_variant(llm_variant):
    llm_to_paligemma = {
            'gemma2_2b': '3b',
            'gemma2_9b': '10b',
            'gemma2_27b': '28b'
        }
    
    if llm_variant not in llm_to_paligemma:
        raise ValueError(f"Unknown LLM variant: {llm_variant}. Supported variants are: {list(llm_to_paligemma.keys())}")
        
    return llm_to_paligemma[llm_variant]


# # Evaluation/inference loop.
def make_predictions(data_iterator, *, num_examples=None,
                     batch_size=4, seqlen, sampler="greedy",data_sharding, params, tokenizer, decode):
  outputs = []
  while True:
    # Construct a list of examples in the batch.
    examples = []
    try:
      for _ in range(batch_size):
        examples.append(next(data_iterator))
        examples[-1]["_mask"] = np.array(True)  # Indicates true example.
    except StopIteration:
      if len(examples) == 0:
        return outputs

    # Not enough examples to complete a batch. Pad by repeating last example.
    while len(examples) % batch_size:
      examples.append(dict(examples[-1]))
      examples[-1]["_mask"] = np.array(False)  # Indicates padding example.

    # Convert list of examples into a dict of np.arrays and load onto devices.
    batch = jax.tree.map(lambda *x: np.stack(x), *examples)
    batch = big_vision.utils.reshard(batch, data_sharding)

    # Make model predictions
    tokens = decode({"params": params}, batch=batch,
                    max_decode_len=seqlen, sampler=sampler)

    # Fetch model predictions to device and detokenize.
    tokens, mask = jax.device_get((tokens, batch["_mask"]))
    tokens = tokens[mask]  # remove padding examples.
    responses = [postprocess_tokens(t, tokenizer) for t in tokens]

    # Append to html output.
    for example, response in zip(examples, responses):
      outputs.append((example["image"], response))
      if num_examples and len(outputs) >= num_examples:
        return outputs

def train_loop(training_config, model, params, tokenizer, data_sharding, update_fn):

      # define decode functions for predictions
    decode_fn = predict_fns.get_all(model)['decode']
    decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())

    train_dataset = big_vision.datasets.jsonl.DataSource(
        os.path.join(training_config.data_dir, "data_train90.jsonl"),
        fopen_keys={"image": training_config.data_dir})

    val_dataset = big_vision.datasets.jsonl.DataSource(
        os.path.join(training_config.data_dir, "data_val10.jsonl"),
        fopen_keys={"image": training_config.data_dir})
    
    print(f'Rendering training examples to HTML')
    # filename = render_examples_to_plot(train_data_iterator(train_dataset, training_config.seq_length, tokenizer), 
    #                                 tokenizer, n_examples=8, 
    #                              filename=os.path.join(training_config.logdir,"training_examples.png"))
    
    filename = render_examples_to_html(train_data_iterator(train_dataset, training_config.seq_length, tokenizer), 
                                    tokenizer=tokenizer, n_examples=8, 
                                 filename=os.path.join(training_config.logdir,"training_examples.html"))
    
    print(f"Done rendering training examples. {filename}")

    TRAIN_STEPS = training_config.train_examples // training_config.batch_size
    EVAL_STEPS = TRAIN_STEPS // 4
    WARMUP_ITER = training_config.warmup_iter
    #EVAL_STEPS = 1
    train_data_it = train_data_iterator(train_dataset, training_config.seq_length, tokenizer)

    sched_fn = big_vision.utils.create_learning_rate_schedule(
        total_steps=TRAIN_STEPS+1, base=training_config.learning_rate,
        decay_type="cosine", warmup_percent=0.10)

    timers = {
        "data_loading": 0.0,
        "forward_backward": 0.0,
        "evaluation": 0.0,
        "other": 0.0,
        "warmup_steps_time":0.0,
    }

    print(f'Starting training with {WARMUP_ITER} warmup iterations')
    # Track total training time excluding warmup and evaluation
    total_training_time = 0
    train_start_time = time.time()

    # chrono setup
    chrono = big_vision.utils.Chrono()
    def simple_measure(name, value):
        print(f"{name}: {value}")

    def simple_measure(name, value):
        print(f"{name}: {value}")

    def write_note(note):
        print(f"NOTE: {note}")
    chrono.inform(measure=simple_measure,write_note=write_note)
    chrono.inform(first_step=1, total_steps=TRAIN_STEPS, global_bs=training_config.batch_size)

    for step in range(1, TRAIN_STEPS+1):
        step_start_time = time.time()
        
        # Make list of N training examples.
        data_start = time.time()
        examples = [next(train_data_it) for _ in range( training_config.batch_size)]

        # Convert list of examples into a dict of np.arrays and load onto devices.
        batch = jax.tree.map(lambda *x: np.stack(x), *examples)
        batch = big_vision.utils.reshard(batch, data_sharding)
        data_time = time.time() - data_start

        # Training step and report training loss
        learning_rate = sched_fn(step)

        fw_bw_start = time.time()
        # if step == 3:
        #      print(f'capturing the trace at step {step}')
        #      prof_step_st = time.time()
        #      jax.profiler.start_trace("./jax_traces/tensorboard/step_3")
        params, loss = update_fn(params, batch, learning_rate)
        # if step == 3:
        #       params = jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
        #       prof_step_et = time.time()
        #       print(f'Done capturing the trace at step {step}')
        #       print(f'Profiler step executed in {prof_step_et-prof_step_st:.4f}')
        #       jax.profiler.stop_trace()
        
        loss = jax.device_get(loss)
        chrono.tick(step)
        fw_bw_time = time.time() - fw_bw_start
        print(f'Step time: {fw_bw_time:.4f}')
        
        
        # Record timings if past warmup
        if step > WARMUP_ITER:
            timers["data_loading"] += data_time
            timers["forward_backward"] += fw_bw_time
            iter_time = time.time() - step_start_time
            total_training_time += iter_time
            timers["other"] += iter_time - (data_time + fw_bw_time)
       
        print(f"step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}")

        
        if (step % 2) == 0:
            print(chrono.note)  # Human-readable progress: ETA, core_hours, etc.
            print(f"Total training time so far: {chrono.accum_train_time:.2f}s")
            print(f"Total evaluation time so far: {chrono.accum_pause_time:.2f}s")
        
        if (step % 123123123) == 0:
            chrono.pause(wait_for=())
            get_device_memory_usage(training_config.device)
            chrono.resume()

        # if (step % 12314123) == 0:
        #     # Pause training time calculation during evaluation
           
        #     print(f"Model predictions at step {step}")
        #     eval_start = time.time()
        #     html_out = ""
        #     for image, caption in make_predictions(
        #         validation_data_iterator(val_dataset,training_config.seq_length, tokenizer), 
        #         num_examples=8, 
        #         batch_size=8,
        #         seqlen=training_config.seq_length,
        #         data_sharding=data_sharding,
        #         params=params,
        #         tokenizer=tokenizer,
        #         decode=decode):
        #         html_out += render_example(image, caption)
        #         # Save the HTML output to a file
        #     outputs_file_name = os.path.join(training_config.logdir,f"predictions_at_{step}.html")
        #     # outputs_file_name = plot_examples(outputs,outputs_file_name)
        #     with open(outputs_file_name, "w") as f:
        #         f.write(html_out)
        #     # Calculate and print evaluation time (not included in training time)
        #     eval_time = time.time() - eval_start
        #     timers['evaluation']+=eval_time
        #     print(f'Predictions at step {step}  took {eval_time:.2f}s saved to {outputs_file_name}')

        if step <= WARMUP_ITER:
            timers['warmup_steps_time'] += time.time() - step_start_time


    print(f"FINAL: Total train time: {chrono.accum_train_time:.2f}s")
    print(chrono.note)  # Human-readable progress: ETA, core_hours, etc.
    # print_and_save_profiling(timers,
    #                         train_start_time,
    #                         total_training_time,
    #                         training_config.logdir)


def transform_update_fn(update_fn, compile_function=False, logdir="logs", 
                         compiled_args=None):
    """
    Analyze memory usage of an update function with options to compile or just annotate.
    
    Args:
        update_fn: The original undecorated function to analyze
        compile_function: If True, lower and compile; if False, return decorated function
        logdir: Directory to save analysis logs
        compiled_args: Dictionary with compilation-specific arguments:
            - params: Model parameters
            - batch: Batch of data for test execution
            - learning_rate: Learning rate to pass to update_fn
            - params_sharding: Sharding spec for parameters
            - data_sharding: Sharding spec for data
        
    Returns:
        Dictionary with function and memory analysis results
    """

    timers = {
        'transform_time':0.0,
        'lowering_time':0.0,
        'compile_time':0.0,
    }

    # Create logs directory if it doesn't exist
    os.makedirs(logdir, exist_ok=True)
    
    # If not compiling, wrap the function with JIT and return directly
    if not compile_function:
        print("Creating JIT-decorated function without compilation")
        
        # Create decorated function with the right annotations
        @functools.partial(jax.jit, donate_argnums=(0,))
        def decorated_fn(params, batch, learning_rate):
            return update_fn(params, batch, learning_rate)
        
        return {
            "fn": decorated_fn,
            "compiled": False,
            "memory_analysis": None
        }
    
    # For compilation path, verify compilation arguments are provided
    if not compiled_args or not isinstance(compiled_args, dict):
        raise ValueError("compiled_args dictionary must be provided when compile_function=True")
    
    # Extract required arguments for compilation
    fake_params = compiled_args.get('fake_params')
    fake_batch = compiled_args.get('fake_batch')
    learning_rate = compiled_args.get('learning_rate', 0.001)  # Default value provided
    params_sharding = compiled_args.get('params_sharding')
    data_sharding = compiled_args.get('data_sharding')
    
    # Validate required arguments
    missing_args = []
    if fake_params is None: missing_args.append('fake_params')
    if fake_batch is None: missing_args.append('fake_batch')
    if params_sharding is None: missing_args.append('params_sharding')
    if data_sharding is None: missing_args.append('data_sharding')
    
    if missing_args:
        raise ValueError(f"Missing required compilation arguments: {', '.join(missing_args)}")
    
    # Print Save Residuals
    save_residual_file = os.path.join(logdir,"update_fn_saved_residuals.txt")
    print(f'Printing Save Residuals to {save_residual_file}')
    with open(save_residual_file, "w") as f:
        with redirect_stdout(f):
            jax.ad_checkpoint.print_saved_residuals(
                update_fn, fake_params, fake_batch, 0.001
            )

    # Save foward and backward JAXPR
    fwd_jaxpr, bwd_jaxpr = fwd_bwd_jaxprs(update_fn, fake_params, fake_batch, 0.001)
    
    # Use proper path joining with logdir
    fwd_jaxpr_path = os.path.join(logdir, "update_fn_jaxpr_fwd.txt")
    bwd_jaxpr_path = os.path.join(logdir, "update_fn_jaxpr_bwd.txt")
    jaxpr_path = os.path.join(logdir, "update_fn_jaxpr.txt")
    
    # Save forward JAXPR
    with open(fwd_jaxpr_path, "w") as f:
        f.write(fwd_jaxpr.pretty_print(use_color=False))
    print(f"Forward JAXPR saved to {fwd_jaxpr_path}")
    
    # Save backward JAXPR
    with open(bwd_jaxpr_path, "w") as f:
        f.write(bwd_jaxpr.pretty_print(use_color=False))
    print(f"Backward JAXPR saved to {bwd_jaxpr_path}")
    
    # Save complete JAXPR
    jaxpr = jax.make_jaxpr(update_fn)(fake_params, fake_batch, 0.001)
    with open(jaxpr_path, "w") as f:
        f.write(jaxpr.pretty_print(use_color=False))
    print(f"Complete JAXPR saved to {jaxpr_path}")

    # For compilation path, create JIT-annotated function with explicit sharding
    print("Creating JIT function with explicit sharding for compilation")

    st = time.time()
    jitted_fn = jax.jit(
        update_fn, 
        donate_argnums=(0,),
        in_shardings=(
            params_sharding,
            data_sharding,
            None  # learning_rate has no sharding
        ),
        out_shardings=(
            params_sharding,
            None  # loss has no sharding
        ),
    )
    timers['transform_time'] = time.time()-st
    print(f'JIT Transformed update_fn in {timers["transform_time"]:.2f}')
    # Lower the function
   
    print(f"Lowering the function with learning rate {learning_rate}...")
    st = time.time()
    lowered_fn = jitted_fn.lower(fake_params, fake_batch, learning_rate)
    timers['lowering_time'] = time.time() - st
    print(f'Lowered update_fn in {timers["lowering_time"]:.2f}')
    
    # Save the lowered function text
    lowered_path = os.path.join(logdir, "update_fn_lowered.txt")
    with open(lowered_path, "w") as f:
        f.write(lowered_fn.as_text())
    print(f"Lowered function saved to {lowered_path}")
    
    # Compile the function
    print("Compiling the function...")
    st = time.time()
    compiled_fn = lowered_fn.compile()
    timers['compile_time'] = time.time()-st
    print(f'Compiled update_fn in {timers["compile_time"]:.2f}')
    
    # Save the compiled function text
    compiled_path = os.path.join(logdir, "update_fn_compiled.txt")
    with open(compiled_path, "w") as f:
        f.write(compiled_fn.as_text())
    print(f"Compiled function saved to {compiled_path}")
    
    # Get memory analysis
    memory_analysis = compiled_fn.memory_analysis()
    
    # Format memory analysis for printing
    device_mem_gb = memory_analysis.temp_size_in_bytes / 1024 / 1024 / 1024
    weight_mem_gb = memory_analysis.argument_size_in_bytes / 1024 / 1024 / 1024
    total_mem_gb = (memory_analysis.argument_size_in_bytes + memory_analysis.temp_size_in_bytes) / 1024 / 1024 / 1024
    host_mem_gb = memory_analysis.host_temp_size_in_bytes / 1024 / 1024 / 1024
    
    print(
        f"Memory Analysis Results:\n"
        f"  Device temp memory: {device_mem_gb:.2f} GB\n"
        f"  Weight memory: {weight_mem_gb:.2f} GB\n"
        f"  Total device memory: {total_mem_gb:.2f} GB\n"
        f"  Host memory: {host_mem_gb:.2f} GB"
    )
    
    # Save memory analysis to file
    memory_path = os.path.join(logdir, "memory_analysis.txt")
    with open(memory_path, "w") as f:
        f.write(f"Device temp memory: {device_mem_gb:.2f} GB\n")
        f.write(f"Weight memory: {weight_mem_gb:.2f} GB\n")
        f.write(f"Total device memory: {total_mem_gb:.2f} GB\n")
        f.write(f"Host memory: {host_mem_gb:.2f} GB\n")
    print(f"Memory analysis saved to {memory_path}")
    
    # Create a simple dictionary with the results
    results = {
        "fn": compiled_fn,
        "compiled": True,
        "memory_analysis": {
            "device_memory_gb": device_mem_gb,
            "weight_memory_gb": weight_mem_gb,
            "total_memory_gb": total_mem_gb,
            "host_memory_gb": host_mem_gb
        }
    }

    print_and_save_compilation_timers(timers,logdir)
    return results
   


def train(model_config, training_config):
    
    # Get the PaliGemma variant from the model config
    paligemma_variant = get_paligemma_variant(model_config.llm.variant)
    print(f"Using PaliGemma variant: {paligemma_variant}")

    model, tokenizer, params = get_model_and_params(model_config,
                                                    training_config,
                                                    paligemma_variant,
                                                    ) 
    
    mesh, data_sharding, params_sharding = get_data_and_param_shardings('1d',params)
    
    params, trainable_mask = get_sharded_trainable_params(
                params,
                params_sharding, 
                log_file=os.path.join(training_config.logdir, "parameter_overview_after_sharding.log")
            )


 
 
    # The main update_fn using a simple stochastic gradient descent (SGD).
    def update_fn(params, batch, learning_rate):
        imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

        def loss_fn(params):
            text_logits, _ = model.apply({"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True)
            logp = jax.nn.log_softmax(text_logits, axis=-1)
            # The model takes as input txts[:, :-1] but the loss is defined as predicting
            # next tokens txts[:, 1:]. Additionally, mask_loss[:, 1:] indicates which tokens
            # are part of the loss (e.g. prefix and padded tokens are not included).
            mask_loss = batch["mask_loss"][:, 1:]
            targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])
            # Compute the loss per example. i.e. the mean of per token pplx.
            # Since each example has a different number of tokens, normalize it.
            token_pplx = jnp.sum(logp * targets, axis=-1)  # sum across vocab_size.
            example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)  # sum across seq_len.
            example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)  # weight by num of tokens.
            # batch_loss: mean of per example loss.
            return jnp.mean(example_loss)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)

            # Apply gradients to trainable params using SGD.
        def apply_grad(param, gradient, trainable):
            if not trainable: return param
            return param - learning_rate * gradient

        params = jax.tree_util.tree_map(apply_grad, params, grads, trainable_mask)

        return params, loss
        
    compiled_args = None
    compile_function = False
    if training_config.compile is True:
        # prepare fake batch, from train iterator
        train_dataset = big_vision.datasets.jsonl.DataSource(
        os.path.join(training_config.data_dir, "data_train90.jsonl"),
        fopen_keys={"image": training_config.data_dir})
        train_data_it =train_data_iterator(train_dataset, training_config.seq_length, tokenizer)
        # Make list of N training examples.
        examples = [next(train_data_it) for _ in range(training_config.batch_size)]
        # Convert list of examples into a dict of np.arrays and load onto devices.
        fake_batch = jax.tree.map(lambda *x: np.stack(x), *examples)
        fake_batch = big_vision.utils.reshard(fake_batch, data_sharding)

        compiled_args={
            'fake_params': params,
            'fake_batch': fake_batch,
            'learning_rate': 0.001,
            'params_sharding': params_sharding,
            'data_sharding': data_sharding
        }
        compile_function = True

    update_fn_transformed = transform_update_fn(
                                update_fn,
                                compile_function=compile_function,
                                compiled_args=compiled_args,
                                logdir=training_config.logdir
                            )['fn']

    train_loop(training_config=training_config,
                model=model,
                params=params,
                tokenizer=tokenizer,
                data_sharding=data_sharding,
                update_fn=update_fn_transformed)

 


   