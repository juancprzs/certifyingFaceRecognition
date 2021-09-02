# Connect
ssh -A -t kw60661.kaust.edu.sa # I dont know why this one shows an error with /tmp/pip-*
ssh -A -t kw60623.kaust.edu.sa # Working. Currently running 1e-3 and 1e-4
ssh -A -t kw60624.kaust.edu.sa # Working
ssh -A -t kw60746.kaust.edu.sa

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

scp -r /home/perezjc/Documents/cert_real_face_perts/weights/ kw60624.kaust.edu.sa:/home/perezjc/cert_real_face_perts

scp -r /home/perezjc/Documents/cert_real_face_perts/venv kw60624.kaust.edu.sa:/home/perezjc/cert_real_face_perts

scp -r data/stylegan_ffhq_recog_png/w.npy kw60624.kaust.edu.sa:/home/perezjc/cert_real_face_perts/data/stylegan_ffhq_recog_png

# FRS model weights
scp -r weights/ kw60624.kaust.edu.sa:/home/perezjc/cert_real_face_perts
# GAN weights
scp -r models/pretrain/ kw60624.kaust.edu.sa:/home/perezjc/cert_real_face_perts/models

