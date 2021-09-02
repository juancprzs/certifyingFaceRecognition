#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[9,99]
#SBATCH -J lr1e-1
#SBATCH -o logs/lr1e-1.%J.out
#SBATCH -e logs/lr1e-1.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

source venv/bin/activate

python main_attack.py \
--output-dir lr1e-1_m0.${SLURM_ARRAY_TASK_ID} \
--lr 1e-1 \
--momentum 0.${SLURM_ARRAY_TASK_ID} \
--loss xent \
--optim SGD
