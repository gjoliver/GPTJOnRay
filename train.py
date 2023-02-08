""""/mnt/shared_storage/jungong/gpt_j/models/models--EleutherAI--gpt-j-6B/snapshots/6e35e2148e92edf096e94d39ac2b98ad59e25975
"""

import argparse
from os import path

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import bitsandbytes as bnb
import pandas as pd
import ray
from ray.air import session, ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def dataset(args):
    """Load training text.
    """
    return ray.data.read_text(args.data_file, drop_empty_lines=False)


def train(args):
    """Load and run the model.
    """
    train_dataset = session.get_dataset_shard("train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        # Add padding token. GPT-J does not seem to have a default
        # padding token.
        # This doesn't really matter because attention_mask should
        # exclude all the padding inputs after tokenization.
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
    )
    model.train()

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.num_epochs):
        for batch in train_dataset.iter_batches(
                batch_size=args.batch_size,
        ):
            optimizer.zero_grad()

            batch = tokenizer(
                batch["text"].values.tolist(),
                padding="longest",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors='pt',
            )
            print(batch)
            outputs = model(**batch)

            loss = F.cross_entropy(
                outputs.logits[:, :-1, :].flatten(0, -2),
                batch['input_ids'][:, 1:].flatten(),
                reduction='mean'
            )

            loss.backward()
            optimizer.step()

            step += 1
            # Report result.
            results = {
                "step": step,
                "loss": loss.detach().item(),
            }
            session.report(results)

    # Save fine-tuned model at output directory.
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=True,
        help="Path to a pretrained huggingface GPT-J model.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        required=True,
        help="Path to a text data file for fine-tuning the model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum training text sequence length.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="Learning rate hyper parameter.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of train epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Train batch size.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Where to save the fine-tuned model.",
    )

    args = parser.parse_args()

    # Actually run.
    trainer = TorchTrainer(
        train,
        train_loop_config=args,
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=1,
            resources_per_worker={
                "GPU": 4,
            },
        ),
        datasets={
            "train": dataset(args),
        },
    )
    result = trainer.fit()

    print(result)
