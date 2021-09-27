#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-24]
#SBATCH -J adam_xent_it5_rest5_lr1e-1
#SBATCH -o logs/adam_xent_it5_rest5_lr1e-1.%J.out
#SBATCH -e logs/adam_xent_it5_rest5_lr1e-1.%J.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

source venv/bin/activate

python main_attack.py \
--chunks 25 --num-chunk ${SLURM_ARRAY_TASK_ID} \
--load-embs --embs-file embs.pth \
--attack-type manual \
--loss xent \
--restarts 5 \
--iters 5 \
--lr 1e-1 \
--output-dir adam_xent_it5_rest5_lr1e-1
