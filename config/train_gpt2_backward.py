
out_dir = 'out-gpt2-backward'
wandb_log = True # override via command line if you like
wandb_project = 'gpt2-backward'
wandb_run_name = 'mini-gpt'
backward = True

compile = True # do not torch compile the model


# ---------------- GPT 2 -----------------------------
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 5 * 8
# this makes total number of tokens be 300B
max_iters = 6001
lr_decay_iters = 600000

# eval stuff
eval_interval = 400
eval_iters = 200
log_interval = 100

# weight decay
weight_decay = 1e-1
# ----------------------------------------