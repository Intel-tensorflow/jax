import os
import jax
import numpy as np
import io
import time
import matplotlib.pyplot as plt
import textwrap
import os
import base64
from PIL import Image
import html
import big_vision.utils
from src.dataset import postprocess_tokens

def get_device_memory_usage(device) -> None:
    num_devices = jax.local_device_count(device)
    gpu_memory_usage = []
    for i in range(num_devices):
        memory_stats = jax.local_devices()[i].memory_stats()
        peak_bytes = memory_stats["peak_bytes_in_use"]
        bytes_limit = memory_stats["bytes_limit"]
        usage_percentage = (peak_bytes / bytes_limit) * 100
        gpu_memory_usage.append((f"{device}{i}", usage_percentage, peak_bytes / (1024 ** 3), bytes_limit / (1024 ** 3)))  # Convert bytes to GB
    
    # Print the results in a tabular format
    print(f"{'Device':<10} {'Usage (%)':<10} {'Peak (GB)':<10} {'Limit (GB)':<10}")
    print("-" * 50)
    for device_name, usage, peak_gb, limit_gb in gpu_memory_usage:
        print(f"{device_name:<10} {usage:<10.2f} {peak_gb:<10.2f} {limit_gb:<10.2f}")

# Print params to show what the model is made of.
def parameter_overview(params, output_file=None):
    """Print an overview of parameters and optionally save to a file."""
    lines = []
    for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
        line = f"{path:80s} {str(arr.shape):22s} {str(arr.sharding)} {arr.dtype}"
        lines.append(line)
        print(line)
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:  # Changed from "w" to "a" for append mode
            f.write("\n".join(lines) + "\n")
        print(f"Parameter overview appended to {output_file}")
    
    return lines

def get_data_and_param_shardings(sharding_strategy, params):
     # Configure mesh based on sharding strategy
    if sharding_strategy == '2d':
        print("Using 2D sharding (data, model)")
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 2), ("data", "model"))
        params_sharding = big_vision.sharding.infer_sharding(
            params, strategy=[('.*', 'fsdp(axis="model")')], mesh=mesh)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("data", None))
    else:
        print("Using 1D sharding (data)")
        mesh = jax.sharding.Mesh(jax.devices(), ("data"))
        params_sharding = big_vision.sharding.infer_sharding(
            params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("data"))
    return mesh, data_sharding, params_sharding



def render_inline(image, resize=(128, 128)):
  """Convert image into inline html."""
  image = Image.fromarray(image)
  image.resize(resize)
  with io.BytesIO() as buffer:
    image.save(buffer, format='jpeg')
    image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
    return f"data:image/jpeg;base64,{image_b64}"

def render_example(image, caption):
  image = ((image + 1)/2 * 255).astype(np.uint8)  # [-1,1] -> [0, 255]
  return f"""
    <div style="display: inline-flex; align-items: center; justify-content: center;">
        <img style="width:128px; height:128px;" src="{render_inline(image, resize=(64,64))}" />
        <p style="width:256px; margin:10px; font-size:small;">{html.escape(caption)}</p>
    </div>
    """

def render_examples_to_html(data_iterator, n_examples=8, tokenizer=None, filename="examples/training_examples.html"):
    """
    Render examples from data iterator and save to an HTML file.
    
    Args:
        data_iterator: Iterator yielding example dicts with 'image' and 'text' keys
        n_examples: Number of examples to display
        filename: Path to save the HTML output
    
    Returns:
        Path to the saved HTML file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    html_out = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Examples</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .container { display: flex; flex-wrap: wrap; }
        </style>
    </head>
    <body>
        <h1>Training Examples</h1>
        <div class="container">
    """
    
    for idx, example in zip(range(n_examples), data_iterator):
        caption = postprocess_tokens(example["text"], tokenizer)  # detokenize model input
        caption = caption[len("caption en\n"):]        # strip prefix
        html_out += render_example(example["image"], caption)
    
    html_out += """
        </div>
    </body>
    </html>
    """
    
    # Save the HTML output to a file
    with open(filename, "w") as f:
        f.write(html_out)
    
    return filename

def fwd_bwd_jaxprs(f, *example_args):
    fwd_jaxpr, (y_shape, res_shape) = jax.make_jaxpr(
        lambda *args: jax.vjp(f, *args), return_shape=True)(*example_args)
    bwd_jaxpr = jax.make_jaxpr(lambda res, outs: res(outs))(res_shape, y_shape)
    return fwd_jaxpr, bwd_jaxpr

def plot_examples(examples, filename="training_examples.png", rows=2, cols=4, figsize=(12, 6)):
    """
    Render image-caption examples using matplotlib and save as PNG.
    
    Args:
        examples: List of (image, caption) tuples
        filename: Output PNG filename
        rows, cols: Grid dimensions
        figsize: Figure size in inches
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (image, caption) in enumerate(examples):
        if i >= len(axes):
            break
            
        # Convert image from [-1,1] to [0,255] uint8
        image_uint8 = ((image + 1)/2 * 255).astype(np.uint8)
        
        # Display image
        axes[i].imshow(image_uint8)
        axes[i].axis('off')
        
        # Wrap caption text to fit in plot
        wrapped_caption = textwrap.fill(caption, width=30)
        axes[i].set_title(wrapped_caption, fontsize=8, wrap=True)
    
    # Hide any unused subplots
    for i in range(len(examples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Examples saved to {filename}")
    return filename

# Process examples and plot them
def render_examples_to_plot(data_iterator, tokenizer, n_examples=8, filename="examples/training_examples.png"):
    """
    Render examples from data iterator to a matplotlib plot and save as PNG.
    
    Args:
        data_iterator: Iterator yielding example dicts with 'image' and 'text' keys
        n_examples: Number of examples to display
        filename: Output PNG filename
    """
    examples = []
    for idx, example in zip(range(n_examples), data_iterator):
        caption = postprocess_tokens(example["text"],tokenizer)  # detokenize model input
        caption = caption[len("caption en\n"):]        # strip prefix
        examples.append((example["image"], caption))
    
    # Calculate grid dimensions
    cols = min(4, n_examples)
    rows = (n_examples + cols - 1) // cols  # ceiling division
    
    return plot_examples(examples, filename=filename, rows=rows, cols=cols)


def print_and_save_profiling(timers, train_start_time, total_training_time, logdir):

      # Calculate combined training time components
    training_components_total = timers['data_loading'] + timers['forward_backward'] + timers['other']
    total_wall_time = time.time() - train_start_time
    
    # Print timing summary to console in tabular format with better alignment
    print(f"\n=== Detailed Timing Breakdown ===")
    print(f"| {'Component':24} | {'Time (s)':10} | {'Percentage':28} |")
    print(f"|{'-'*26}|{'-'*12}|{'-'*30}|")
    print(f"| {'Data loading':24} | {timers['data_loading']:10.2f} | {timers['data_loading']/total_training_time*100:10.1f}% of training time |")
    print(f"| {'Forward/backward pass':24} | {timers['forward_backward']:10.2f} | {timers['forward_backward']/total_training_time*100:10.1f}% of training time |")
    print(f"| {'Other operations':24} | {timers['other']:10.2f} | {timers['other']/total_training_time*100:10.1f}% of training time |")
    print(f"|{'-'*26}|{'-'*12}|{'-'*30}|")
    print(f"| {'Training time':24} | {training_components_total:10.2f} | {training_components_total/total_training_time*100:10.1f}% of training time |")
    #print(f"| {'Pure training time':24} | {total_training_time:10.2f} | {'100.0% of training time':28} |")
    print(f"|{'-'*26}|{'-'*12}|{'-'*30}|")
    print(f"| {'Warmup time':24} | {timers['warmup_steps_time']:10.2f} | {'(not included in training)':28} |")
    print(f"| {'Evaluation time':24} | {timers['evaluation']:10.2f} | {'(not included in training)':28} |")
    print(f"|{'-'*26}|{'-'*12}|{'-'*30}|")
    print(f"| {'Total wall clock time':24} | {total_wall_time:10.2f} | {'-':28} |")
    
    # Save timing summary to file in tabular format with better alignment
    timing_file = os.path.join(logdir, "timing_breakdown.txt")
    mode = "a" if os.path.exists(timing_file) else "w"
    with open(timing_file, mode) as f:
        f.write("\n\n=== Detailed Training Time  Breakdown ===\n\n")
        f.write(f"| {'Component':24} | {'Time (s)':10} | {'Percentage':28} |\n")
        f.write(f"|{'-'*26}|{'-'*12}|{'-'*30}|\n")
        f.write(f"| {'Data loading':24} | {timers['data_loading']:10.2f} | {timers['data_loading']/total_training_time*100:10.1f}% of training time |\n")
        f.write(f"| {'Forward/backward pass':24} | {timers['forward_backward']:10.2f} | {timers['forward_backward']/total_training_time*100:10.1f}% of training time |\n")
        f.write(f"| {'Other operations':24} | {timers['other']:10.2f} | {timers['other']/total_training_time*100:10.1f}% of training time |\n")
        f.write(f"|{'-'*26}|{'-'*12}|{'-'*30}|\n")
        f.write(f"| {'Training time':24} | {training_components_total:10.2f} | {training_components_total/total_training_time*100:10.1f}% of training time |\n")
        #f.write(f"| {'Pure training time':24} | {total_training_time:10.2f} | {'100.0% of training time':28} |\n")
        f.write(f"|{'-'*26}|{'-'*12}|{'-'*30}|\n")
        f.write(f"| {'Warmup time':24} | {timers['warmup_steps_time']:10.2f} | {'(not included in training)':28} |\n")
        f.write(f"| {'Evaluation time':24} | {timers['evaluation']:10.2f} | {'(not included in training)':28} |\n")
        f.write(f"|{'-'*26}|{'-'*12}|{'-'*30}|\n")
        f.write(f"| {'Total wall clock time':24} | {total_wall_time:10.2f} | {'-':28} |\n")
    
    print(f"Timing breakdown saved to {timing_file}")

def print_and_save_compilation_timers(timers, logdir):
    """
    Print and save compilation timing information in a formatted table.
    
    Args:
        timers: Dictionary containing compilation timing metrics
        logdir: Directory to save the timing information
    """
    # Calculate total time
    total_time = sum(timers.values())
    
    # Print timing summary to console in tabular format
    print(f"\n=== Compilation Timing Breakdown ===")
    print(f"| {'Component':24} | {'Time (s)':10} | {'Percentage':28} |")
    print(f"|{'-'*26}|{'-'*12}|{'-'*30}|")
    
    for name, time_value in timers.items():
        percentage = (time_value / total_time * 100) if total_time > 0 else 0
        print(f"| {name:24} | {time_value:10.2f} | {percentage:13.1f}% of total time |")
    
    print(f"|{'-'*26}|{'-'*12}|{'-'*30}|")
    print(f"| {'Total compilation time':24} | {total_time:10.2f} | {'100.0% of total time':28} |")
    
    # Save timing summary to file in tabular format
    timing_file = os.path.join(logdir, "timing_breakdown.txt")
    with open(timing_file, "w") as f:
        f.write("=== Compilation Timing Breakdown ===\n\n")
        f.write(f"| {'Component':24} | {'Time (s)':10} | {'Percentage':28} |\n")
        f.write(f"|{'-'*26}|{'-'*12}|{'-'*30}|\n")
        
        for name, time_value in timers.items():
            percentage = (time_value / total_time * 100) if total_time > 0 else 0
            f.write(f"| {name:24} | {time_value:10.2f} | {percentage:13.1f}% of total time |\n")
        
        f.write(f"|{'-'*26}|{'-'*12}|{'-'*30}|\n")
        f.write(f"| {'Total compilation time':24} | {total_time:10.2f} | {'100.0% of total time':28} |\n")
    
    print(f"Compilation timing breakdown saved to {timing_file}")