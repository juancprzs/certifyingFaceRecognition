#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=0-249:10
#SBATCH -J searchLossAndLr_rest5_it10_lossxent_lr1e+2
#SBATCH -o logs/searchLossAndLr_rest5_it10_lossxent_lr1e+2.%J.out
#SBATCH -e logs/searchLossAndLr_rest5_it10_lossxent_lr1e+2.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

source venv/bin/activate

python main_attack.py \
--chunks 250 --num-chunk ${SLURM_ARRAY_TASK_ID} \
--load-embs --embs-file embs.pth --iters 10 --restarts 5 \
--lr 1e+2 \
--loss xent \
--output-dir searchLossAndLr_rest5_it10_lossxent_lr1e+2
