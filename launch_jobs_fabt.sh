#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-249]
#SBATCH -J fabt_it5_rest5_targ5
#SBATCH -o logs/fabt_it5_rest5_targ5.%J.out
#SBATCH -e logs/fabt_it5_rest5_targ5.%J.err
#SBATCH --time=1:00:00
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
--restarts 5 \
--iters 5 \
--n-target-classes 5 \
--output-dir fabt_it5_rest5_targ5