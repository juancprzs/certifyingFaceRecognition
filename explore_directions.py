import matplotlib
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from proj_utils import get_proj_mat, get_full_points, mvee


PLOT = False
BOUNDARIES_DIR = 'boundaries'
ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
DATASET = ['ffhq', 'celebahq'][0]
GAN_NAME = ['stylegan', 'pggan'][0]
FILE_TEMPLATE = f'boundaries/{GAN_NAME}_{DATASET}_%s_w_boundary.npy'

def plot_inner_prods(dirs):
    inn_prods = np.round(dirs.T @ dirs, 3)

    # Inspired by
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, ax = plt.subplots()
    im = ax.imshow(inn_prods)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(ATTRS))); ax.set_yticks(np.arange(len(ATTRS)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(ATTRS, fontsize=24)
    ax.set_yticklabels(ATTRS, fontsize=24)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ATTRS)):
        for j in range(len(ATTRS)):
            text = ax.text(j, i, inn_prods[i, j], ha="center", va="center", 
                           color="w", fontsize=18)

    ax.set_title("Inner products", fontsize=26)
    fig.tight_layout()
    plt.show()


all_bounds = glob(f'{BOUNDARIES_DIR}/*.npy')

dirs = []
for att in ATTRS:
    this_file = FILE_TEMPLATE % att
    assert this_file in all_bounds, f'Boundary for attribute "{att}" not found!'
    dirs.append(np.load(this_file))

dirs = np.concatenate(dirs, axis=0).T
# dirs.shape == [n_dims, n_dirs] == [n_dims, len(ATTRS)]
assert dirs.shape[1] == len(ATTRS)

if PLOT:
    plot_inner_prods(dirs)

proj_mat = get_proj_mat(dirs) # proj_mat.shape == [n_dims, n_dims]
dirs_expanded, exp_out = get_full_points(dirs, fill_with_null=True)
X, c = mvee(dirs_expanded.T) # Their way
import pdb; pdb.set_trace()
assert np.allclose(c, 0), "The origin should be the ellipses's center"

