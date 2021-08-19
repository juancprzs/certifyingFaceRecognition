source venv/bin/activate

LATENT_CODE_NUM=10; 
ATTR=eyeglasses; 
LAT=w; 
FF=results/stylegan_ffhq_"$ATTR"_editing_"$LAT"/; 
rm -rf $FF; 
CUDA_VISIBLE_DEVICES=0 python edit.py -m stylegan_ffhq -b boundaries/stylegan_ffhq_"$ATTR"_w_boundary.npy -n "$LATENT_CODE_NUM" -o "$FF" -s "$LAT" -i data/stylegan_ffhq/"$LAT".npy;