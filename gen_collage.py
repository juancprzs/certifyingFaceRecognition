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
lims = dir_name.replace('/', '').split('_')[-2:]
lo, hi = float(lims[0]), float(lims[1])
all_ims = glob(f'{dir_name}/*_*.jpg')
# Get how many
n_ids = len(set([osp.basename(x).split('_')[0] for x in all_ims]))
n_ims = len(glob(f'{dir_name}/000_*.jpg'))
titls = np.linspace(lo, hi, n_ims)

fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111, nrows_ncols=(n_ids, n_ims), 
    axes_pad=(0.05, 0.1)) # inches pad b/w axes in inch.
for idx in range(n_ids):
    these_ims = glob(f'{dir_name}/{idx:03d}_*.jpg')
    these_ims = sorted(these_ims, 
        key=lambda x: int(osp.basename(x).split('_')[-1].split('.')[0]))
    assert len(these_ims) == n_ims
    for jdx, im_path in enumerate(these_ims):
        img = mpimg.imread(im_path)
        tot_idx = int(idx*n_ims + jdx)
        grid[tot_idx].imshow(img)
        grid[tot_idx].axis('off')
        if idx == 0: # Add title on first row
            num = titls[jdx]
            titl = f'{num:3.2f}' if num < 0 else f'+{num:3.2f}'
            grid[tot_idx].set_title(titl, fontsize=26)
    
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
