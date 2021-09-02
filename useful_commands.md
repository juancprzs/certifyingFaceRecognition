# Copy stuff
scp -r /home/perezjc/Documents/cert_real_face_perts/weights/ kw60661.kaust.edu.sa:/home/perezjc/cert_real_face_perts
scp -r /home/perezjc/Documents/cert_real_face_perts/venv kw60661.kaust.edu.sa:/home/perezjc/cert_real_face_perts

# Connect
ssh -A -t kw60661.kaust.edu.sa
ssh -A -t kw60623.kaust.edu.sa

# Clone repo and configure for pull
git config --global credential.helper store
git clone https://github.com/juancprzs/cert_real_face_perts.git

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


conda env create -f base_juan.yml
conda activate base_juan
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install facenet-pytorch
mkdir -p data/stylegan_ffhq_recog_png/
scp -r data/stylegan_ffhq_recog_png/w.npy kw60623.kaust.edu.sa:/home/perezjc/cert_real_face_perts/data/stylegan_ffhq_recog_png
scp -r weights/ kw60623.kaust.edu.sa:/home/perezjc/cert_real_face_perts
scp -r models/pretrain/ kw60623.kaust.edu.sa:/home/perezjc/cert_real_face_perts/models

