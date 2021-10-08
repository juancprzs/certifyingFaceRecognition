#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=0-249:10
#SBATCH -J search_adam_diff_lr1e+1
#SBATCH -o logs/search_adam_diff_lr1e+1.%J.out
#SBATCH -e logs/search_adam_diff_lr1e+1.%J.err
#SBATCH --time=0:20:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

source venv/bin/activate

python main_attack.py \
--chunks 250 --num-chunk ${SLURM_ARRAY_TASK_ID} \
--load-embs --embs-file embs.pth --iters 20 --restarts 10 \
--lr 1e+1 \
--loss diff \
--output-dir search_adam_diff_lr1e+1
