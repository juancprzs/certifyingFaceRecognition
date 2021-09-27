#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-249]
#SBATCH -J fabt_it10_rest10_targ10
#SBATCH -o logs/fabt_it10_rest10_targ10.%J.out
#SBATCH -e logs/fabt_it10_rest10_targ10.%J.err
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
--load-embs --embs-file embs.pth \
--attack-type fab-t \
--restarts 10 \
--iters 10 \
--n-target-classes 10 \
--output-dir fabt_it10_rest10_targ10
