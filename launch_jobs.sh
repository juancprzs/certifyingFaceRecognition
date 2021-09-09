#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-24]
#SBATCH -J adam_xent
#SBATCH -o logs/adam_xent.%J.out
#SBATCH -e logs/adam_xent.%J.err
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
--output-dir debug_adam_lr1e-0 \
--lr 1e-0 \
--loss xent
