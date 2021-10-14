#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=0-499
#SBATCH -J certification
#SBATCH -o logs/certification.%J.out
#SBATCH -e logs/certification.%J.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

source venv/bin/activate

instance=${SLURM_ARRAY_TASK_ID};
skip=$((1+$instance));
maxx=$((2*$skip));

python certify.py \
--face-recog-model insightface \
--outfile instance_$instance.txt \
--sigma 1e-2 \
--embs-file embs.pth \
--skip $skip \
--max $maxx
