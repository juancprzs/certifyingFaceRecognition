cd Documents/cert_real_face_perts/
source venv/bin/activate
# Original one
LATENT_CODE_NUM=11; LAT=w; 
ATTR=smile;
LIM=0.8; FF=results/stylegan_ffhq_"$ATTR"_editing_"$LAT"_-"$LIM"_+"$LIM"/; 
rm -rf $FF; CUDA_VISIBLE_DEVICES=0 python edit.py \
-m stylegan_ffhq \
-b boundaries/stylegan_ffhq_"$ATTR"_w_boundary.npy \
--steps "$LATENT_CODE_NUM" \
-o "$FF" \
-s "$LAT" \
-i data/stylegan_ffhq/"$LAT".npy \
--start_distance -"$LIM" \
--end_distance +"$LIM"; python gen_collage_explore_lims.py -i "$FF"


# Ours
SAMPLES=66; # = 6*11 
ATTR=eyeglasses; 
LAT=w; 
FF=results/stylegan_ffhq_"$ATTR"_editing_"$LAT"/; 
rm -rf $FF; CUDA_VISIBLE_DEVICES=0 python edit_constrained_perturbed.py \
-m stylegan_ffhq \
--samples "$SAMPLES" \
-o "$FF" \
-s "$LAT" \
-i data/stylegan_ffhq/"$LAT".npy; python gen_collage_samples.py -i "$FF"
