#!/bin/bash

# Training
python3 ./train.py ./config/train_gpt2.py --out_dir=gpt2-fw-sgd-1339 --wandb_run_name=gpt2-lg-owt-sgd-1339 --init_from=scratch --eval_interval=300 --eval_iters=200 --eval_only=False

# Merge model from merge_dir into model from resume_dir at stitch_layer_index i
# We can choose whether to preserve the original head or not
for i in {1..11}
do
python3 ./train.py ./config/train_gpt2.py --out_dir=out --resume_dir=gpt2-fw-adam-1337 --merge_dir=gpt2-fw-sgd-1337 --wandb_run_name=gpt2-owt-l_$i-adam-sgd-adam_h --init_from=merge --stitch_layer_index=$i --eval_interval=200 --eval_iters=200 --use_original_head=True --eval_only=True
done

# Stitch model from merge_dir into model from resume_dir at stitch_layer_index i
# We can choose whether to preserve the original head or not
for i in {1..11}
do
python3 ./train.py ./config/train_gpt2.py --out_dir=out --resume_dir=gpt2-fw-sgd-1337 --merge_dir=gpt2-fw-sgd-1339 --wandb_run_name=gpt2-owt-l_$i-sgd-sgd2_h --init_from=stitch --stitch_layer_index=$i --eval_interval=200 --eval_iters=200 --use_original_head=True
done