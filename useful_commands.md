# Connect
ssh -A -t kw60661.kaust.edu.sa # I dont know why this one shows an error with /tmp/pip-*
ssh -A -t kw60828.kaust.edu.sa # For EGO4d (no slurm)
ssh -A -t kw60746.kaust.edu.sa # Doesnt recognize password

ssh -A -t kw60624.kaust.edu.sa # Working. Currently running 1e-1 and 1e-0
ssh -A -t kw60623.kaust.edu.sa # Slurm not working
ssh -A -t kw60749.kaust.edu.sa # Slurm not working



# Anaconda if it doesnt exist
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh

# Clone repo and configure for pull
git config --global credential.helper store
git clone https://github.com/juancprzs/cert_real_face_perts.git

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

# Environment things
cd cert_real_face_perts
conda update -n base -c defaults conda
conda env create -f base_juan.yml
# conda activate base_juan
# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install facenet-pytorch
mkdir -p data/stylegan_ffhq_recog_png/
mkdir logs

# FRS model weights
scp -r /home/perezjc/Documents/cert_real_face_perts/weights/ kw60749.kaust.edu.sa:/home/perezjc/cert_real_face_perts

# The environment
scp -r /home/perezjc/Documents/cert_real_face_perts/venv kw60749.kaust.edu.sa:/home/perezjc/cert_real_face_perts

# The w codes
scp -r /home/perezjc/Documents/cert_real_face_perts/data/stylegan_ffhq_recog_png/w.npy kw60749.kaust.edu.sa:/home/perezjc/cert_real_face_perts/data/stylegan_ffhq_recog_png

# GAN weights
scp -r /home/perezjc/Documents/cert_real_face_perts/models/pretrain/ kw60749.kaust.edu.sa:/home/perezjc/cert_real_face_perts/models

The nice chunk is 12 from 25
Or, equivalently, chunks 120-129 from 250


# Generate embeddings
We run
frs_method=insightface; python main_attack.py --embs-file embeddings/embs_1M_"$frs_method".pth --restarts 1 --iters 1 --output-dir compute_1M_embs_"$frs_method" --face-recog-method $frs_method