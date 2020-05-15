#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
# source activate py13-cuda9
source activate pt11-cuda9

# ===============================================================
# GLOVE emb
path_vocab=lib/vocab/clctotal+swbd.min-count4.en

# ---- [no dd] ----
# fname=test_clc_valid
# ftst=lib-bpe/clc/nobpe/clc-valid.src
# seqlen=150

# fname=test_clc
# ftst=lib-bpe/clc/nobpe/clc-test.src
# seqlen=125

# fname=test_nict
# ftst=lib-bpe/nict/nobpe/nict.src
# seqlen=85

# fname=test_dtal
# ftst=lib-bpe/dtal/nobpe/dtal.src
# seqlen=165

# fname=test_eval3 # default segauto
# ftst=lib-bpe/eval3/nobpe/eval3.src
# seqlen=145

# ---- [after dd] ----
# fname=test_clc_valid_afterdd
# ftst=lib-bpe/after-ddcls-glove/clc_valid.txt
# seqlen=150

# fname=test_clc_afterdd
# ftst=lib-bpe/after-ddcls-glove/clc.txt
# seqlen=125

# fname=test_nict_afterdd
# ftst=lib-bpe/after-ddcls-glove/nict.txt
# seqlen=85

fname=test_dtal_afterdd
ftst=lib-bpe/after-ddcls-glove/dtal.txt
seqlen=165

# fname=test_eval3_afterdd # default segauto
# ftst=lib-bpe/after-ddcls-glove/eval3.txt
# seqlen=145

# ==================================================================
# BPE emb
# path_vocab=lib-bpe/vocab/bpe_en_25000+pad.txt

# ---- [no dd] ----
# fname=test_clc_valid
# ftst=lib-bpe/clc/valid.src
# seqlen=155

# fname=test_clc
# ftst=lib-bpe/clc/test.src
# seqlen=135

# fname=test_nict
# ftst=lib-bpe/nict/nict.src
# seqlen=90

# fname=test_dtal
# ftst=lib-bpe/dtal/dtal.src
# seqlen=170

# fname=test_eval3 # default segauto
# ftst=lib-bpe/eval3/eval3.src
# seqlen=145

# ---- [after dd] ----
# fname=test_oet_afterdd
# ftst=lib/oet-asr/eval-v3/dd/oet.ddres.aln.flt.split.bpe
# seqlen=265

# fname=test_clc_valid_afterdd
# ftst=lib-bpe/after-ddcls-bpe/clc_valid.txt.bpe
# seqlen=155

# fname=test_clc_afterdd
# ftst=lib-bpe/after-ddcls-bpe/clc.txt.bpe
# seqlen=135

# fname=test_nict_afterdd
# ftst=lib-bpe/after-ddcls-bpe/nict.txt.bpe
# seqlen=90

# fname=test_dtal_afterdd
# ftst=lib-bpe/after-ddcls-bpe/dtal.txt.bpe
# seqlen=170

# fname=test_eval3_afterdd # default segauto
# ftst=lib-bpe/after-ddcls-bpe/eval3.txt.bpe
# seqlen=145


# ----- models ------
model=models-v6new/debug-v001s
ckpt=2020_05_15_00_27_55

# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/grammatical-error-correction/translate.py \
    --test_path_src $ftst \
    --test_path_tgt $ftst \
    --path_vocab_src $path_vocab \
    --path_vocab_tgt $path_vocab \
    --load $model/checkpoints/$ckpt \
    --test_path_out $model/$fname/$ckpt/ \
    --max_seq_len $seqlen \
    --batch_size 50 \
    --use_gpu True \
    --beam_width 1 \
    --seqrev False \
    --eval_mode 2
