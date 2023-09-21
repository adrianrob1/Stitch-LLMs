This project wouldn't have been possible without [nanoGPT](https://github.com/karpathy/nanoGPT).

# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `pip install transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast BPE code <3
- `pip install wandb` for optional logging <3
- `pip install tqdm` <3

## quick start

### Train Model
python3 ./train.py ./config/train_gpt2.py

### Merge models
Merge model from merge_dir into model from resume_dir at stitch_layer_index i.   
We can choose whether to preserve the original head or not.

python3 ./train.py ./config/train_gpt2.py --out_dir=out --resume_dir=gpt2-fw-adam-1337 --merge_dir=gpt2-fw-sgd-1337 --wandb_run_name=gpt2-owt-l_$i-adam-sgd-adam_h --init_from=merge --stitch_layer_index=$i --eval_interval=200 --eval_iters=200 --use_original_head=True --eval_only=True

### Stitch models
Stitch model from merge_dir into model from resume_dir at stitch_layer_index i.   
We can choose whether to preserve the original head or not.

python3 ./train.py ./config/train_gpt2.py --out_dir=out --resume_dir=gpt2-fw-adam-1337 --merge_dir=gpt2-fw-sgd-1337 --wandb_run_name=gpt2-owt-l_$i-adam-sgd-adam_h --init_from=merge --stitch_layer_index=$i --eval_interval=200 --eval_iters=200 --use_original_head=True
### Sample
python3 ./sample.py --out_dir=out-gpt2-backward