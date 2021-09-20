#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=30g
#SBATCH --cpus-per-task=10
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

taskid=${SLURM_ARRAY_TASK_ID}

DATE=`date +%Y%m%d`

declare -a fold_list=(0 1 2 3 4)
arglen=${#fold_list[@]}
i=$(( taskid%arglen ))

data_bin=wikitext-103-heldout600
stdout="stdout.log"
validate_interval=1000
save_interval_updates=5000



SAVE="checkpoint/wikitext-103-lm/${DATE}/${data_bin}"

TENSORBOARD=${SAVE}/tensorboard

rm -r ${SAVE}; mkdir -p ${SAVE} ${TENSORBOARD}

python train.py --task language_modeling \
    data-bin/${data_bin} \
    --save-dir ${SAVE} \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 6 --tokens-per-sample 3072 --seed 1 \
    --log-interval 100 --log-format simple \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
    --no-last-checkpoints \
    --save-interval-updates ${save_interval_updates} --keep-interval-updates 1 \
    --validate-interval ${validate_interval} --no-epoch-checkpoints \
    | tee -a ${SAVE}/${stdout}

