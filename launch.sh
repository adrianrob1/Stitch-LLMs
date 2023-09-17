#!/bin/bash

#for i in {1..11}
#do

#python3 ./train.py ./config/train_gpt2.py --out_dir=out --resume_dir=gpt2-fw-adam-1337 --merge_dir=gpt2-fw-sgd-1337 --wandb_run_name=gpt2-owt-l_$i-adam-sgd-adam_h --init_from=merge --stitch_layer_index=$i --eval_interval=200 --eval_iters=200 --use_original_head=True --eval_only=True

#done

for i in {1..11}
do

python3 ./train.py ./config/train_gpt2.py --out_dir=out --resume_dir=gpt2-fw-sgd-1337 --merge_dir=gpt2-fw-sgd-1339 --wandb_run_name=gpt2-owt-l_$i-sgd-sgd2_h --init_from=merge --stitch_layer_index=$i --eval_interval=200 --eval_iters=200 --use_original_head=True --eval_only=True

done

#for i in {1..5}
#do

#python3 ./train.py ./config/train_shakespeare_char.py --out_dir=out --resume_dir=out-gpt2-char-shakespeare-adam --merge_dir=out-gpt2-char-shakespeare-adam2 --wandb_run_name=gpt2-shakespeare-l$i-adam-adam2 --init_from=merge --stitch_layer_index=$i --eval_interval=200 --eval_iters=200  --use_original_head=False --eval_only=True

#done
