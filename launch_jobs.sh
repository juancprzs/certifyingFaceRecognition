#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-5]
#SBATCH -J adam_xent
#SBATCH -o logs/adam_xent.%J.out
#SBATCH -e logs/adam_xent.%J.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

source venv/bin/activate

python main_attack.py \
--output-dir adam_lr1e-${SLURM_ARRAY_TASK_ID} \
--lr 1e-${SLURM_ARRAY_TASK_ID} \
--loss xent
