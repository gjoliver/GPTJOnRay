# GPTJOnRay

This example shows how to fine-tune a GPT-J-6B model using Ray Air
and HuggingFace Accelerate on a single g5.12xlarge instance with
4 A10G GPUs.

Example texts generated after 1 epoch of fine-tuning using the sample Shakespeare dataset:

```
prompt: I am
gpt-j: I am not mad, I have a daughter, call her forth,<|endoftext|>

prompt: to be
gpt-j: to be a man, and to be a woman too, and to be a<|endoftext|>

prompt: honor
gpt-j: honor, and the best of my blood, and the heir of<|endoftext|>
```

### Step 1
Download and cache a pre-trained gpt-j-6B model locally.
The examples uses [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B).

To learn how to download a HuggingFace model, refer to
[this script](https://github.com/gjoliver/DreamBoothOnRay/blob/master/cache_model.py).

### Step 2
Fine tune.
```
python train.py \
    --model_dir=<local model cache dir> \
    --output_dir=<dir to save the tuned model> \
    --data_file=<file containing training texts>
```
Please check [train.py](https://github.com/gjoliver/GPTJOnRay/blob/master/train.py)
for other command line arguments that can be used to configure the training run.

This repo provides a sample data file containing all of Shakespeare's plays.

### Genereate text
To run the fine-tuned gpt-j model.
```
python run.py --model_dir=<output dir from step 2>
```
