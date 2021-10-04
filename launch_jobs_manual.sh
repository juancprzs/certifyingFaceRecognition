#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=0-249:25
#SBATCH -J adam_diff_it50_rest100_lr1e-0
#SBATCH -o logs/adam_diff_it50_rest100_lr1e-0.%J.out
#SBATCH -e logs/adam_diff_it50_rest100_lr1e-0.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

source venv/bin/activate

python main_attack.py \
--chunks 250 --num-chunk ${SLURM_ARRAY_TASK_ID} \
--load-embs --embs-file embs.pth \
--output-dir adam_diff_it50_rest100_lr1e-0_newinitonsurf
