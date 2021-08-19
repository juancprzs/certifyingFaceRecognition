from glob import glob
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid

N_IDS = 6
N_IMS = 10

latent = 'z'
attribute = 'eyeglasses'
dir_name = f'results/stylegan_ffhq_{attribute}_editing_{latent}'

fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111, nrows_ncols=(N_IDS, N_IMS), 
    axes_pad=(0.05, 0.1)) # inches pad b/w axes in inch.
for idx in range(N_IDS):
    these_ims = glob(f'{dir_name}/{idx:03d}_*.jpg')
    these_ims = sorted(these_ims, 
        key=lambda x: int(osp.basename(x).split('_')[-1].split('.')[0]))
    assert len(these_ims) == N_IMS
    for jdx, im_path in enumerate(these_ims):
        img = mpimg.imread(im_path)
        tot_idx = int(idx*N_IMS + jdx)
        grid[tot_idx].imshow(img)
        grid[tot_idx].axis('off')
    
plt.show()
