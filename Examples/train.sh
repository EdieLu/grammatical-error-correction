#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
# source activate pt11-cuda9
source activate py13-cuda9

# ------------------------ DIR --------------------------
savedir=models-v6new/debug-v001/
use_bpe=False
train_path_src=lib-bpe/clc/nobpe/clc-train.src
train_path_tgt=lib-bpe/clc/nobpe/clc-train.tgt
dev_path_src=lib-bpe/clc/nobpe/clc-valid.src
dev_path_tgt=lib-bpe/clc/nobpe/clc-valid.tgt
# dev_path_src=None
# dev_path_tgt=None
path_vocab=lib/vocab/clctotal+swbd.min-count4.en
load_embedding=lib/embeddings/glove.6B.200d.txt

# ------------------------ TRAIN --------------------------
checkpoint_every=5
print_every=1
batch_size=256
max_seq_len=32
num_epochs=20
random_seed=300
eval_with_mask=True

# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/grammatical-error-correction/train.py \
	--train_path_src $train_path_src \
	--train_path_tgt $train_path_tgt \
	--dev_path_src $dev_path_src \
	--dev_path_tgt $dev_path_tgt \
	--path_vocab_src $path_vocab \
	--path_vocab_tgt $path_vocab \
	--load_embedding_src $load_embedding \
	--load_embedding_tgt $load_embedding \
	--use_bpe $use_bpe \
	--save $savedir \
	--random_seed $random_seed \
	--embedding_size_enc 200 \
	--embedding_size_dec 200 \
	--hidden_size_enc 200 \
	--num_bilstm_enc 2 \
	--num_unilstm_enc 0 \
	--hidden_size_dec 200 \
	--num_unilstm_dec 4 \
	--hidden_size_att 10 \
	--hard_att False \
	--att_mode bilinear \
	--residual True \
	--hidden_size_shared 200 \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--batch_first True \
	--seqrev False \
	--eval_with_mask $eval_with_mask \
	--scheduled_sampling False \
	--teacher_forcing_ratio 1.0 \
	--dropout 0.2 \
	--embedding_dropout 0.0 \
	--num_epochs $num_epochs \
	--use_gpu True \
	--learning_rate 0.001 \
	--max_grad_norm 1.0 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--additional_key_size 0 \
