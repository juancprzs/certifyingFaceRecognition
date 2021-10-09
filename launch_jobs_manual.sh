#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=0-249:10
#SBATCH -J search_debug_notsurf_xent_lr1e+1
#SBATCH -o logs/search_debug_notsurf_xent_lr1e+1.%J.out
#SBATCH -e logs/search_debug_notsurf_xent_lr1e+1.%J.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

source venv/bin/activate

python main_attack.py \
--chunks 250 --num-chunk ${SLURM_ARRAY_TASK_ID} \
--load-embs --embs-file embs.pth --iters 20 --restarts 20 \
--not-on-surf \
--lr 1e+1 \
--loss xent \
--output-dir search_debug_notsurf_xent_lr1e+1
