import torch
import os.path as osp
from glob import glob
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


def evaluate(exp_name, log_files, data_files):
    print(f'Evaluating based on these {len(log_files)} files: ', log_files)
    tot_instances, tot_successes, tot_magnitudes = 0, 0, 0.
    for log_file in log_files:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        data = { l.split(':')[0] : float(l.split(':')[1]) for l in lines }
        tot_instances += int(data.pop('instances'))
        curr_successes = data.pop('successes')
        tot_successes += int(curr_successes)
        curr_magnitude = float(data.pop('avg_mags'))
        tot_magnitudes += curr_magnitude * float(curr_successes)


    attack_success_rate = 100.*float(tot_successes) / tot_instances
    avg_magnitude = tot_magnitudes / tot_successes if tot_successes != 0 else 0

    # Evaluate the found deltas
    print(f'Computing delta stats based on these {len(data_files)} files: ', 
        data_files)
    deltas = [torch.load(x, map_location='cpu')['deltas'] for x in data_files]
    deltas = torch.cat(deltas)
    magnitudes = [torch.load(x, map_location='cpu')['magnitudes'] for x in data_files]
    magnitudes = torch.cat(magnitudes)

    # Compare all radii with the magnitudes of the found deltas
    dists = magnitudes.sqrt()
    N = dists.size(0)
    maxx = torch.quantile(dists, 0.99).item() # 99th quantile
    lins = torch.linspace(0, maxx, N)
    all_comps = dists.unsqueeze(1) > lins.unsqueeze(0)
    # Do the counts
    counts = all_comps.sum(0)
    # Normalize the counts
    norm_counts = counts / tot_instances
    # Plot everything
    avg_dist = dists.mean().item()
    label = f'{exp_name} (avg. pert.$={avg_dist:3.2f}$)'
    plt.plot(lins, norm_counts, label=label)
    plt.grid(True)


EXPS = {
    'FaceNet$^C$'   : 'FABT_METHOD_facenet-N_100000',
    'FaceNet$^V$'   : 'FABT_METHOD_facenet-vggface2-N_100000',
    'ArcFace'       : 'FABT_METHOD_insightface-N_100000'
}

for exp_name, exp in EXPS.items():
    log_files = glob(osp.join('exp_results', exp, 'logs', '*.txt'))
    data_files = glob(osp.join('exp_results', exp, 'results', '*.pth'))
    evaluate(exp_name, log_files, data_files)

plt.xlabel(r'$\|\delta\|_{\Sigma,2}$', fontsize=16)
plt.ylabel(r'Accuracy', fontsize=16)
plt.legend()
plt.title('Accuracy \\textit{vs.} perturbation budget', fontsize=20)

figname = 'FAB-attacks-compare.png'
plt.savefig(figname, dpi=200)

