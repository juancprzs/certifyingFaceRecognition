import torch
import numpy as np
import pandas as pd
import os.path as osp
from glob import glob
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

N_INSTANCES = 500
RESULTS_DIR = 'cert_results'
SIGMAS = ['1e-1', '2.5e-1', '5e-1', '7.5e-1', '1e+0']
x_label = 'Radius'
METHODS = {
    'insightface'       : 'ArcFace',
    'facenet'           : 'FaceNet$^C$', 
    'facenet-vggface2'  : 'FaceNet$^V$',
}
# First isotropic certification
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
all_max_radii = {}
for idx, (method, nice_name) in enumerate(METHODS.items()):
    max_radii = np.zeros(N_INSTANCES)
    for sigma in SIGMAS:
        sigma_dir = f'SAMPLE_METHOD_{method}-N_100000-SIGMA_{sigma}'
        txt_files = glob(osp.join(RESULTS_DIR, sigma_dir, 'instance_*.txt'))
        print(f'Directory {sigma_dir} has {len(txt_files)} instances')
        assert len(txt_files) == N_INSTANCES
        all_dfs = []
        for txt_file in txt_files:
            curr_df = pd.read_csv(txt_file, sep='\t')
            assert curr_df.shape[0] == 1 # the DF has one row of data
            all_dfs.append(curr_df)

        df = pd.concat(all_dfs)
        df = df.sort_values('idx')
        radii = df['radius'].values
        N = radii.shape[0]
        assert N == N_INSTANCES
        maxx = radii.max()
        lins = np.linspace(0, maxx, N)
        all_comps = np.expand_dims(radii, 1) > np.expand_dims(lins, 0)
        counts = all_comps.sum(0)
        # Normalize the counts
        norm_counts = counts / N

        # Keep track of largest radii
        where_larger = radii > max_radii
        max_radii[where_larger] = radii[where_larger]
    
    # Plot the current envelope
    lins = np.linspace(0, max_radii.max(), N)
    all_comps = np.expand_dims(max_radii, 1) > np.expand_dims(lins, 0)
    counts = all_comps.sum(0)
    norm_counts = counts / N
    axes[idx].plot(lins, norm_counts, label='Isotropic envelope')
    # Save envelope for last plot
    all_max_radii[method] = max_radii
    # Plot details
    axes[idx].legend(fontsize=16)
    axes[idx].set_title(nice_name, fontsize=20)
    axes[idx].set_xlabel(x_label, fontsize=16)
    if idx == 0: axes[idx].set_ylabel('Certified accuracy', fontsize=16)
    axes[idx].grid(True)


# Then anisotropic certification

ANIS = True
EQUIV_RAD = ANIS # True
if ANIS:
    SIGMAS.extend(['2.5e+0']) # , '5e+0', '7.5e+0'])
    if EQUIV_RAD:
        from scipy.special import gamma
        from attack_utils.gen_utils import get_all_matrices
        # This (the *inverse) is equivalent to the sigma for smoothing
        sigma_mat = get_all_matrices()[6].cpu()
        n = sigma_mat.size(0)
        def get_vol_coeff(mod_sigma_mat):
            pi_part = np.sqrt(np.pi**n)
            gamma_part = gamma(n/2 + 1)
            det = torch.prod(mod_sigma_mat)
            coeff = (pi_part / gamma_part * det) ** (1/n)
            return coeff.item()
        
        x_label = 'Equivalent-volume radius'

all_max_radii = {}
for idx, (method, nice_name) in enumerate(METHODS.items()):
    max_radii = np.zeros(N_INSTANCES)
    for sigma in SIGMAS:
        sigma_dir = f'SAMPLE_METHOD_{method}-N_100000-SIGMA_{sigma}'
        if ANIS:
            sigma_dir = 'ANIS_' + sigma_dir
            if EQUIV_RAD:
                mod_sigma_mat = float(sigma) * sigma_mat
        
        txt_files = glob(osp.join(RESULTS_DIR, sigma_dir, 'instance_*.txt'))
        print(f'Directory {sigma_dir} has {len(txt_files)} instances')
        assert len(txt_files) == N_INSTANCES
        all_dfs = []
        for txt_file in txt_files:
            curr_df = pd.read_csv(txt_file, sep='\t')
            assert curr_df.shape[0] == 1 # the DF has one row of data
            all_dfs.append(curr_df)

        df = pd.concat(all_dfs)
        df = df.sort_values('idx')
        if ANIS and EQUIV_RAD:
            radii = get_vol_coeff(mod_sigma_mat) * df['gap'].values
        else:
            radii = df['radius'].values

        N = radii.shape[0]
        assert N == N_INSTANCES
        maxx = radii.max()
        lins = np.linspace(0, maxx, N)
        all_comps = np.expand_dims(radii, 1) > np.expand_dims(lins, 0)
        counts = all_comps.sum(0)
        # Normalize the counts
        norm_counts = counts / N

        # Keep track of largest radii
        where_larger = radii > max_radii
        max_radii[where_larger] = radii[where_larger]
    
    # Plot the current envelope
    lins = np.linspace(0, max_radii.max(), N)
    all_comps = np.expand_dims(max_radii, 1) > np.expand_dims(lins, 0)
    counts = all_comps.sum(0)
    norm_counts = counts / N
    axes[idx].plot(lins, norm_counts, label='Anisotropic envelope')
    # Save envelope for last plot
    all_max_radii[method] = max_radii
    # Plot details
    axes[idx].legend(fontsize=16)
    axes[idx].set_title(nice_name, fontsize=20)
    axes[idx].set_xlabel(x_label, fontsize=16)
    if idx == 0: axes[idx].set_ylabel('Certified accuracy', fontsize=16)
    axes[idx].grid(True)

figname = 'anisotropic-isotropic-comparison.png'

print(f'Saving figure at {figname}')
plt.savefig(figname, dpi=200, bbox_inches='tight')
