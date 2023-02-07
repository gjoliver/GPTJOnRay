""""/mnt/shared_storage/jungong/gpt_j/models/models--EleutherAI--gpt-j-6B/snapshots/6e35e2148e92edf096e94d39ac2b98ad59e25975
"""

import argparse
from os import path

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def run(args):
    """Load and run the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Create a model and initialize it with empty weights
    config = AutoConfig.from_pretrained(args.model_dir)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Load the checkpoint and dispatch it to the right devices
    model = load_checkpoint_and_dispatch(
        model,
        path.join(args.model_dir, "pytorch_model.bin"),
        device_map="auto",
        no_split_module_classes=["GPTJBlock"],
    )

    while True:
        x = input("----- prompt: ")
        if x == "exit": break

        inputs = tokenizer(x, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"])

        print(tokenizer.decode(outputs[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=True,
        help="Path to a pretrained huggingface GPT-J model.",
    )

    args = parser.parse_args()

    # Actually run.
    run(args)
