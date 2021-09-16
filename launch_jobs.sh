#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-24]
#SBATCH -J sgd_xent_lr1e+2_m99
#SBATCH -o logs/sgd_xent_lr1e+2_m99.%J.out
#SBATCH -e logs/sgd_xent_lr1e+2_m99.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

source venv/bin/activate

python main_attack.py \
--chunks 25 \
--num-chunk ${SLURM_ARRAY_TASK_ID} \
--optim SGD \
--momentum 0.99 \
--output-dir sgd_xent_lr1e+2_m99 \
--lr 1e+2 \
--loss xent