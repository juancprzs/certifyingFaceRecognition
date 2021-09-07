import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchvision.transforms import Compose, Normalize
from ..proj_utils import (get_projection_matrices, sample_ellipsoid,
    project_to_region_pytorch, set_seed, in_subs, in_ellps)

# Script argument-dependent constants
INP_RESOLS = { 'insightface' : 112, 'facenet' : 160 }
STD = 0.5
MEAN = 0.5


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


def eval_files(files, final_log):
    print(f'Evaluating based on the following {len(files)} files: ', files)
    tot_instances = 0
    tot_corr = {}
    for log_file in files:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        data = { l.split(':')[0] : float(l.split(':')[1]) for l in lines }
        instances = int(data.pop('n_instances'))
        tot_instances += instances
        for atck, acc in data.items():
            corr = acc * instances
            if atck in tot_corr:
                tot_corr[atck] += corr
            else:
                tot_corr[atck] = corr

    accs = {atck : float(corr)/tot_instances for atck, corr in tot_corr.items()}
    accs.update({ 'n_instances' : tot_instances })
    info = '\n'.join([f'{k}:{v}' if k == 'n_instances' else f'{k}:{v:4.2f}'
        for k, v in accs.items()])
    print_to_log(info, final_log)
    print(f'Saved all results to {final_log}')


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
    

def lat2embs(net, lat_codes, transform, few=False, with_tqdm=False):
    all_ims, all_embs = [], []
    itr = net.get_batch_inputs(lat_codes)
    if with_tqdm: itr = tqdm(itr)
    for batch in itr:
        with torch.set_grad_enabled(few):
            # Forward and store
            ims = net.easy_synthesize(batch.to(DEVICE), **KWARGS)['image']
            all_ims.append(ims if few else ims.detach().cpu())
            # Transform, forward and store
            ims = TRANSFORM(ims)
            ims = ims.unsqueeze(0) if ims.ndim == 3 else ims
            embs = net(ims)
            all_embs.append(embs if few else embs.detach().cpu())

    return torch.cat(all_embs), torch.cat(all_ims)


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


def find_adversaries(net, lat_codes, labels, orig_embs, optim_name, lr, iters,
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
        embs, _ = lat2embs(net, lat_codes + deltas, transform, few=True)
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


def eval_chunk(model, dataset, batch_size, chunks, num_chunk, device, args):
    testloader, start_ind, end_ind = get_data_utils(dataset, batch_size, chunks, 
                                                    num_chunk)
    # Clean acc
    clean_acc = get_clean_acc(model, testloader, device)
    # Compute adversarial instances
    advs, labels = compute_advs(model, testloader, device, batch_size, 
                                args.cheap, args.seed, args.eps)
    # Compute robustness
    accs = compute_adv_accs(model, advs, labels, device, batch_size)
    # Send everything to file
    accs.update({'clean' : clean_acc , 'n_instances' : len(testloader.dataset)})
    log_file = save_results(advs, labels, accs, args, num_chunk, start_ind,
                            end_ind)

    return log_file


def save_results(deltas, successes, num_chunk, output_dir):
    torch.save(deltas, osp.join(output_dir, f'deltas_{num_chunk}.pth'))
    torch.save(successes, osp.join(output_dir, 
        f'successes_{num_chunk}.pth'))


def eval_chunk(net, num_chunk, DEVICE, args):
    args.LOGGER.info(f'Processing chunk {num_chunk} out of {args.chunks}')
    chunk_length = len(LAT_CODES) / args.chunks
    # Ensure chunk length is valid
    assert chunk_length.is_integer(), 'Partition of set should be exact'
    chunk_length = int(chunk_length)
    assert chunk_length % net.batch_size == 0, 'Batch size MUST be preserved'
    # Extract the actual chunk
    start = num_chunk * chunk_length
    lat_cods_chunk = LAT_CODES[start:(start+chunk_length)]
    # Iterate through batches
    n_succ, tot = 0, 0
    deltas, successes = [], []
    transform = get_transform(INP_RESOLS[args.face_recog_method], MEAN, STD)
    for idx, btch_cods in enumerate(tqdm(net.get_batch_inputs(lat_cods_chunk))):
        # 'Compute' the labels: indices in original array
        batch_size = btch_cods.size(0)
        labels = torch.arange(idx*batch_size, (idx+1)*batch_size, device=DEVICE)
        labels += start # The offset induces by chunking
        # The actual computation of the adversaries
        curr_deltas, succ = find_adversaries(
            net, btch_cods.to(DEVICE), labels, embs, optim_name=args.optim, 
            lr=args.lr, iters=args.iters, momentum=args.momentum, 
            frs_method=args.face_recog_method, loss_type=args.loss, 
            transform=transform, random_init=True, 
            rand_init_on_surf=not args.not_on_surf
        )
        tot += batch_size
        n_succ += succ.sum()
        pbar.set_description(f'-> {n_succ} adversaries for {tot} identities')
        # Append results
        successes.append(succ)
        deltas.append(curr_deltas)

    deltas, successes = torch.cat(deltas), torch.cat(successes)
    # Store adversaries and the successes
    save_results(deltas, successes, num_chunk, output_dir)