# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too often
max_iters = 5000

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 320 # context of up to 256 previous characters

# baby GPT model: iter 5000: loss 0.8183, time 7645.87ms, mfu 6.52%
# number of parameters: 10.65M
# n_layer = 6
# n_head = 6
# n_embd = 384
# dropout = 0.2

n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.2

learning_rate = 4e-4 # with baby networks can afford to go a bit higher
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 4e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
