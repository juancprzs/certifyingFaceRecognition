import argparse
import numpy as np
from glob import glob
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid

parser = argparse.ArgumentParser(
    description='Generate collage of images.')
parser.add_argument('-i', '--input_dir', type=str, required=True,
                    help='Directory to save the output results. (required)')
args = parser.parse_args()

dir_name = args.input_dir
# dir_name is of the form
# 'results/stylegan_ffhq_eyeglasses_editing_w_-3_+3/'
# We need to extract the limits: -3 and +3
all_ims = glob(f'{dir_name}/*_*.jpg')
# Get how many
n_ids = len(set([osp.basename(x).split('_')[0] for x in all_ims]))
n_ims = len(glob(f'{dir_name}/000_*.jpg'))

for idx in range(n_ids):
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_ims//11, n_ims//6), 
        axes_pad=(0.05, 0.1)) # inches pad b/w axes in inch.
    these_ims = glob(f'{dir_name}/{idx:03d}_*.jpg')
    assert len(these_ims) == n_ims
    for jdx, im_path in enumerate(these_ims):
        img = mpimg.imread(im_path)
        tot_idx = int(jdx)
        grid[tot_idx].imshow(img)
        grid[tot_idx].axis('off')
    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    plt.close()
