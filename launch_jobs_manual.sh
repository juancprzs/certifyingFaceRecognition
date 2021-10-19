#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=0-249
#SBATCH -J insightface_5k
#SBATCH -o logs/insightface_5k.%J.out
#SBATCH -e logs/insightface_5k.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

# source venv/bin/activate

FRS_METHOD=insightface
N_EMBS=5000
EXP_DIR=METHOD_$FRS_METHOD-N_$N_EMBS

python main_attack.py \
--not-on-surf \
--load-embs --chunks 50000 --num-chunk ${SLURM_ARRAY_TASK_ID} \
--load-n-embs $N_EMBS \
--face-recog-method $FRS_METHOD \
--output-dir $EXP_DIR

python main_attack.py --eval-files --output-dir $EXP_DIR
