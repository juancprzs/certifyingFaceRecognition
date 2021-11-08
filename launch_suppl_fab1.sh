#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=0-249
#SBATCH -J fabt_suppl_it1_rest1_100k
#SBATCH -o logs/fabt_suppl_it1_rest1_100k.%J.out
#SBATCH -e logs/fabt_suppl_it1_rest1_100k.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

IT=1
REST=1

source venv/bin/activate

FRS_METHOD=insightface
N_EMBS=100000
EXP_DIR=SUPPL-FABT-METHOD_$FRS_METHOD-N_$N_EMBS-IT_$IT-REST_$REST

python main_attack.py \
--attack-type fab-t --load-embs --num-chunk ${SLURM_ARRAY_TASK_ID} \
--load-n-embs $N_EMBS \
--face-recog-method $FRS_METHOD \
--output-dir $EXP_DIR \
--n-target-classes 5 \
--iters $IT \
--restarts $REST
