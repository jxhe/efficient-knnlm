#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:4
#SBATCH --mem=50g
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=compute-1-18
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15



DATE=`date +%Y%m%d`
dataset='law'
max_tokens=1024
tokens_per_sample=512
mode="eos"
update_freq=2
dropout=0.1

validate_interval=1000
save_interval_updates=5000

SAVE="checkpoint/${dataset}/${DATE}"

ckpt="wmtnc_lm_ckpt/wmt19.en/model.pt"

rm -rf ${SAVE}; mkdir -p ${SAVE}

ln ${ckpt} ${SAVE}/checkpoint_load.pt

python train.py --task language_modeling \
    data-bin/${dataset} \
    --arch transformer_lm_gbw \
    --save-dir ${SAVE} \
    --restore-file ${SAVE}/checkpoint_load.pt \
    --max-update 30000 --max-lr 1.0 --t-mult 2 --lr-period-updates 27000 --lr-scheduler cosine --lr-shrink 0.6 \
    --warmup-updates 4000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --decoder-embed-dim 1536 --decoder-input-dim 1024 --decoder-output-dim 1024 \
    --decoder-layers 20 --decoder-ffn-embed-dim 6144 \
    --criterion cross_entropy --max-tokens ${max_tokens} --update-freq ${update_freq} --tokens-per-sample ${tokens_per_sample} --seed 1 \
    --log-interval 100 --log-format simple \
    --sample-break-mode ${mode} --skip-invalid-size-inputs-valid-test \
    --no-last-checkpoints \
    --reset-meters --reset-optimizer --reset-lr-scheduler \
    --save-interval-updates ${save_interval_updates} --keep-interval-updates 1 \
    --validate-interval ${validate_interval} --no-epoch-checkpoints \
    --fp16 \
    | tee -a ${SAVE}/stdout.log
