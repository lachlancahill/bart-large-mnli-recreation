import os
import subprocess
import sys
from transformers import AutoModelForSequenceClassification, AutoConfig


def load_deepspeed_checkpoint(checkpoint_dir, config, device_map, torch_dtype):
    # Define the paths
    zero_to_fp32_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    output_file = os.path.join(checkpoint_dir, "pytorch_model.bin")

    # Check if the pytorch_model.bin already exists
    if not os.path.exists(output_file):
        # Run the zero_to_fp32.py script to convert the checkpoint using the current Python interpreter
        subprocess.run(
            [sys.executable, zero_to_fp32_script,
             checkpoint_dir,
             output_file],
            check=True
        )

    # Load the model using the converted checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        config=config,
        device_map=device_map,
        torch_dtype=torch_dtype
    )

    return model