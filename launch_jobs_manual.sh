#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=120-129
#SBATCH -J small_search_notsurf_adam_it1_rest100_lr1e+0
#SBATCH -o logs/small_search_notsurf_adam_it1_rest100_lr1e+0.%J.out
#SBATCH -e logs/small_search_notsurf_adam_it1_rest100_lr1e+0.%J.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

source venv/bin/activate

python main_attack.py \
--chunks 250 --num-chunk ${SLURM_ARRAY_TASK_ID} \
--load-embs --embs-file embs.pth --iters 1 --restarts 100 \
--not-on-surf \
--lr 1e+0 \
--loss xent \
--output-dir small_search_notsurf_adam_it1_rest100_lr1e+0
