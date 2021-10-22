#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=0-50%23
#SBATCH -J cert_insightface_sigma_1e-2
#SBATCH -o logs/cert_insightface_sigma_1e-2.%J.out
#SBATCH -e logs/cert_insightface_sigma_1e-2.%J.err
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account conf-cvpr-2021.11.23-ghanembs

source venv/bin/activate

FRS_METHOD=insightface
N_EMBS=100000
SIGMA=1e-2
EXP_DIR=certif_results/sigma_$SIGMA/METHOD_$FRS_METHOD-N_$N_EMBS

instance=${SLURM_ARRAY_TASK_ID};
skip=$((1+$instance));
maxx=$((2*$skip));

python certify.py \
--skip $skip --max $maxx \
--outfile $EXP_DIR/instance_$instance.txt \
--load-n-embs $N_EMBS \
--face-recog-model $FRS_METHOD \
--sigma $SIGMA

