import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam
from torchvision.transforms import Compose, Normalize
from .proj_utils import (get_projection_matrices, sample_ellipsoid,
    project_to_region_pytorch, set_seed, in_subs, in_ellps)

# Script argument-dependent constants
INP_RESOLS = { 'insightface' : 112, 'facenet' : 160 }
STD = 0.5
MEAN = 0.5
EMB_SIZE = 512
LAT_SPACE = 'w'
DATASET = 'ffhq'
IMAGE_EXT = 'png'
GAN_NAME = 'stylegan'
DEVICE = torch.device('cuda')
OPTIMS = ['Adam', 'SGD']
FRS_METHODS = ['insightface', 'facenet']
KWARGS = { 'latent_space_type' : LAT_SPACE }
LOSS_TYPES = ['away', 'nearest', 'diff', 'xent']
ORIG_DATA_PATH = f'data/{GAN_NAME}_{DATASET}_recog_png'
ORIG_IMAGES_PATH = osp.join(ORIG_DATA_PATH, 'ims')
ORIG_TENSORS_PATH = osp.join(ORIG_DATA_PATH, 'tensors')
LAT_CODES_PATH = osp.join(ORIG_DATA_PATH, f'{LAT_SPACE}.npy')

PROJ_MAT, ELLIPSE_MAT, DIRS, _ = get_projection_matrices(dataset=DATASET, 
    gan_name=GAN_NAME)
DIRS = torch.tensor(DIRS, dtype=torch.float32, device=DEVICE)
PROJ_MAT = torch.tensor(PROJ_MAT, dtype=torch.float32, device=DEVICE)
ELLIPSE_MAT = torch.tensor(ELLIPSE_MAT, dtype=torch.float32, device=DEVICE) 
assert PROJ_MAT.shape[0] == PROJ_MAT.shape[1] == EMB_SIZE
assert ELLIPSE_MAT.shape[0] == ELLIPSE_MAT.shape[1] == EMB_SIZE


def get_latent_codes(generator):
    lat_codes = generator.preprocess(np.load(LAT_CODES_PATH), **KWARGS)
    return torch.from_numpy(lat_codes)


def get_pairwise_dists(embs1, embs2, method='insightface'):
    if method == 'insightface':
        diff = embs1 - embs2
        return torch.norm(diff, dim=1)
    else:
        dot = (embs1 * embs2).sum(dim=1)
        return 1 - dot


def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)


def print_training_params(args, txt_file_path):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    # Print to log and console
    print_to_log(text, txt_file_path)
    print(text)


def args2text(args):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    return text


def get_transform(img_size, mean, std):
    def resize(x):
        x = x.unsqueeze(0) if x.ndim != 4 else x
        interp = F.interpolate(x, size=(img_size, img_size), mode='bilinear', 
            align_corners=False)
        return interp.squeeze(0)

    normalize = Normalize((mean, mean, mean), (std, std, std))
    return Compose([resize, normalize])


def get_optim(deltas, optim_name='SGD', lr=0.001, momentum=0.9):
    if optim_name == 'SGD':
        optim = SGD([deltas], lr=lr, momentum=momentum)
    elif optim_name == 'Adam':
        optim = Adam([deltas], lr=lr)

    return optim


def get_dists(embs1, embs2, method='insightface'):
    if method == 'insightface':
        # Use special compute mode for numerical stability
        return torch.cdist(embs1, embs2, 
            compute_mode='donot_use_mm_for_euclid_dist')
    else:
        return 1 - torch.matmul(embs1, embs2.T)
    

def lat2embs(generator, net, lat_codes, transform, few=False, with_tqdm=False):
    all_ims, all_embs = [], []
    itr = generator.get_batch_inputs(lat_codes)
    if with_tqdm: itr = tqdm(itr)
    print('Before the for-loop')
    for batch in itr:
        with torch.set_grad_enabled(few):
            # Forward and store
            ims = generator.easy_synthesize(batch.to(DEVICE), **KWARGS)['image']
            all_ims.append(ims if few else ims.detach().cpu())
            # Transform, forward and store
            ims = transform(ims)
            ims = ims.unsqueeze(0) if ims.ndim == 3 else ims
            embs = net(ims)
            all_embs.append(embs if few else embs.detach().cpu())

    print('Concat of all embs')
    all_embs = torch.cat(all_embs)
    print('Done concat of embs')

    print('Concat of all ims')
    all_ims = torch.cat(all_ims)
    print('Done concat of ims')
    return all_embs, all_ims


def compute_loss(all_dists, labels, loss_type='away', use_probs=True, 
        scale_dists=True):
    if use_probs:
        if scale_dists:
            all_dists = all_dists / np.sqrt(EMB_SIZE)
        vals = F.softmax(-all_dists, dim=1)
    else:
        vals = all_dists
    
    # -----------   Get val to target
    target_val = torch.gather(vals, 1, labels.view(-1, 1))
    # -----------   Get val to highest-performing class != target
    value = -1 if use_probs else float('inf')
    mod_vals = torch.scatter(vals, 1, labels.view(-1, 1), value)
    if use_probs: # If they're probabilities, we are looking for the max
        nearest_val, _ = torch.max(mod_vals, 1)
    else: # If they're distances, we are looking for the min
        nearest_val, _ = torch.min(mod_vals, 1)

    # -----------   The losses themselves
    # --> 'away' loss
    if loss_type == 'away':
        if use_probs: # If it's a probability, we want to MINIMIZE it
            coeff = +1.
        else: # Otherwise we want to MAXIMIZE it
            coeff = -1.
        return coeff * target_val.mean()
    # --> 'nearest' loss
    if loss_type == 'nearest':
        if use_probs: # If it's a probability, we want to MAXIMIZE it
            coeff = -1.
        else: # Otherwise we want to MAXIMIZE it
            coeff = +1.
        return coeff * nearest_val.mean()
    # --> 'diff' loss
    if loss_type == 'diff':
        diff = target_val - nearest_val
        if use_probs: # If it's a probability, we want to MINIMIZE it
            coeff = +1.
        else: # Otherwise we want to MAXIMIZE it
            coeff = -1.
        return coeff * diff.mean()
    # --> 'xent' loss
    if loss_type == 'xent':
        assert use_probs, 'xent loss should be used together with probs'
        if scale_dists:
            scores = -all_dists / np.sqrt(EMB_SIZE)
        else:
            scores = -all_dists
        xent = F.cross_entropy(input=scores, target=labels, 
            reduction='none')
        # It's the cross-entropy loss, so we want to maximize it
        return -1. * xent.mean()


def find_adversaries(generator, net, lat_codes, labels, orig_embs, optim_name, lr, iters,
        momentum, frs_method, loss_type, transform, random_init=True, 
        rand_init_on_surf=True):
    # Initialize deltas
    if random_init:
        # Sample from ellipsoid and project
        deltas = sample_ellipsoid(ELLIPSE_MAT, n_vecs=lat_codes.size(0))
        deltas, _ = project_to_region_pytorch(deltas, PROJ_MAT, 
            ELLIPSE_MAT, check=True, dirs=DIRS, on_surface=rand_init_on_surf)
        # Transform to tensor
        deltas = deltas.clone().detach().requires_grad_(True)
    else:
        deltas = torch.zeros_like(lat_codes, requires_grad=True)
    
    # Their corresponding optimizer
    optim = get_optim(deltas, optim_name=optim_name, lr=lr, momentum=momentum)
    for idx_iter in range(iters):
        # Get embeddings from perturbed latent codes
        embs, _ = lat2embs(generator, net, lat_codes + deltas, transform, 
            few=True)
        # Compute current predictions and check for attack success
        all_dists = get_dists(embs, orig_embs, method=frs_method)
        preds = torch.argmin(all_dists, 1)
        success = preds != labels
        if torch.all(success): break
        # Compute loss
        loss = compute_loss(all_dists, labels, loss_type=loss_type)
        # Backward
        optim.zero_grad()
        loss.backward()
        # Optim step without forgetting previous values
        old_deltas = torch.clone(deltas).detach()
        optim.step()
        # Projection
        with torch.no_grad():
            deltas, _ = project_to_region_pytorch(deltas, PROJ_MAT, ELLIPSE_MAT, 
                check=True, dirs=DIRS, on_surface=False)
            # Re-establish old values for deltas that were already succesful
            deltas[success] = old_deltas[success]

    # Final check of deltas
    assert in_subs(deltas.T, PROJ_MAT) and in_ellps(deltas.T, ELLIPSE_MAT)

    return deltas.detach().cpu(), success


def get_curr_preds(generator, net, embs, curr_lats, deltas, successes, 
        transform, device, args):
    n_succ = successes.sum()
    # Compute the adversarial images according to the computed deltas
    adv_lat_cds = curr_lats[successes] + deltas[successes]
    # Pad input (if necessary)
    to_pad = generator.batch_size - adv_lat_cds.size(0) % generator.batch_size
    pad_inp = torch.cat([adv_lat_cds, torch.zeros(to_pad, EMB_SIZE)], dim=0)
    adv_embs, adv_ims = lat2embs(generator, net, pad_inp, transform, 
        with_tqdm=True)
    # Extract (since we padded)
    adv_embs, adv_ims = adv_embs[:n_succ], adv_ims[:n_succ]
    # Get the current predictions according to the distances
    curr_dists = get_dists(adv_embs.to(device), embs, 
        method=args.face_recog_method)
    curr_preds = torch.argmin(curr_dists, 1)
    return adv_embs, adv_ims, curr_preds


def check_advs(labels, curr_preds, successes, args):
    # Check where the predictions differ from the labels
    where_adv = labels != curr_preds
    if not torch.all(where_adv):
        args.LOGGER.info('=====> Something is wrong with the adversaries!!!')
        return False
    else:
        n_succ = successes.sum()
        n_ids = successes.size(0)
        args.LOGGER.info(f'Found {n_succ} advs for {n_ids} IDs')
        return True


def save_results(results, deltas, successes, num_chunk, args):
    filename = f'results_chunk{num_chunk}of{args.chunks}'
    # Save the deltas and successes
    data_file = osp.join(args.results_dir, f'{filename}.pth')
    if successes.sum() != 0:
        data = { 
            'deltas' : deltas[successes].detach(), 
            'successes' : torch.nonzero(successes).detach() 
        }
        torch.save(data, data_file)
        flag = 'WAS'
    else:
        flag = 'NOT'
    # Save the log
    log_file = osp.join(args.logs_dir, f'{filename}.txt')
    info = '\n'.join([f'{k}:{v}' for k, v in results.items()]) # Create string
    print_to_log(info, log_file)

    print(f'Evaluation for chunk {num_chunk} out of {args.chunks} finished.\n'
          f'==> Data {flag} saved to {data_file}.\n'
          f'==> Log file saved to {log_file}.\n'
          + 50 * '-' + '\n')
    
    return log_file


def eval_files(log_files, args):
    print(f'Evaluating based on these {len(log_files)} files: ', log_files)
    tot_instances, tot_successes = 0, 0
    for log_file in log_files:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        data = { l.split(':')[0] : float(l.split(':')[1]) for l in lines }
        tot_instances += int(data.pop('instances'))
        tot_successes += int(data.pop('successes'))

    attack_success_rate = 100.*float(tot_successes) / tot_instances
    info = f'successes:{tot_successes}\n' \
        f'instances:{tot_instances}\n' \
        f'rate:{attack_success_rate:4.2f}'

    print_to_log(info, args.final_results)
    args.LOGGER.info(f'Saved all results to {args.final_results}')


def eval_chunk(generator, net, lat_codes, ims, embs, transform, num_chunk, 
        device, args):
    args.LOGGER.info(f'Processing chunk {num_chunk} out of {args.chunks}')
    chunk_length = len(lat_codes) / args.chunks
    # Ensure chunk length is valid
    assert chunk_length.is_integer(), 'Partition of set should be exact'
    chunk_length = int(chunk_length)
    assert chunk_length % generator.batch_size == 0, \
        f'Batch size MUST be preserved: chunk length={chunk_length} and ' \
        f'batch size={generator.batch_size}'
    # Extract the actual chunk
    start = num_chunk * chunk_length
    lat_cods_chunk = lat_codes[start:(start+chunk_length)]
    ims_chunk = ims[start:(start+chunk_length)]
    # Iterate through batches
    deltas, successes, all_labels = [], [], []
    pbar = tqdm(generator.get_batch_inputs(lat_cods_chunk))
    n_succ, tot = 0, 0
    for idx, btch_cods in enumerate(pbar):
        # 'Compute' the labels: indices in original array
        batch_size = btch_cods.size(0)
        begin = idx * batch_size # Current index times the batch size
        labels = torch.arange(begin, begin+batch_size, device=device)
        labels += start # The offset induced by chunking
        all_labels.append(labels)
        # The actual computation of the adversaries
        curr_deltas, succ = find_adversaries(
            generator, net, btch_cods.to(device), labels, embs, 
            optim_name=args.optim, lr=args.lr, iters=args.iters, 
            momentum=args.momentum, frs_method=args.face_recog_method, 
            loss_type=args.loss, transform=transform, random_init=True, 
            rand_init_on_surf=not args.not_on_surf
        )
        n_succ += succ.sum()
        tot += batch_size
        pbar.set_description(f'-> {n_succ} adversaries for {tot} identities')
        # Append results
        successes.append(succ)
        deltas.append(curr_deltas)

    all_labels = torch.cat(all_labels)
    deltas, successes = torch.cat(deltas), torch.cat(successes)
    n_succ = successes.sum()

    # Check the results
    if n_succ == 0: # No successes
        args.LOGGER.info('Didnt find any adversary! =(')
    else:
        # Get the networks output on the current adversarial examples
        adv_embs, adv_ims, curr_preds = get_curr_preds(generator, net, embs, 
            lat_cods_chunk, deltas, successes, transform, device, args)
        # Check they are indeed adversarial
        assert check_advs(labels, curr_preds, successes, args)
        # Store adversaries and the successes
        # Extract the original images and the identities with which the people 
        # are being confused
        orig, confu = ims_chunk[successes], ims[curr_preds]
        # Plot the images and their adversaries
        plot_advs(orig, adv_ims, confu, args)
    
    # Log the results
    results = { 'successes' : n_succ, 'instances' :  len(all_labels) }
    log_file = save_results(results, deltas, successes, num_chunk, args)
    return log_file

    
def plot_advs(orig, adv_ims, confu, args):
    # Show in a plot
    args.LOGGER.info(f'Plotting adversaries')
    for idx, (ori, adv, conf) in enumerate(zip(orig, adv_ims, confu)):
        plt.figure()
        plt.subplot(131); plt.imshow(ori.cpu().permute(1, 2, 0).numpy())
        plt.axis('off'); plt.title('Original')

        plt.subplot(132); plt.imshow(adv.cpu().permute(1, 2, 0).numpy())
        plt.axis('off'); plt.title('Adversary')

        plt.subplot(133); plt.imshow(conf.cpu().permute(1, 2, 0).numpy())
        plt.axis('off'); plt.title('Prediction')

        plt.tight_layout()
        path = osp.join(args.figs_dir, f'id_{idx}.png')
        plt.savefig(path, bbox_inches='tight', dpi=400)
        plt.close()
    
