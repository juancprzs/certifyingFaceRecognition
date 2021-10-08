import torch
import numpy as np
from tqdm import tqdm
from time import time
import os.path as osp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from autoattack import AutoAttack
from torch.optim import SGD, Adam, RMSprop
from torchvision.transforms import Compose, Normalize
from .proj_utils import (get_projection_matrices, sample_ellipsoid, set_seed,
    proj2region, in_subs, in_ellps, sq_distance)

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
OPTIMS = ['Adam', 'SGD', 'RMSProp']
FRS_METHODS = ['insightface', 'facenet']
KWARGS = { 'latent_space_type' : LAT_SPACE }
LOSS_TYPES = ['away', 'nearest', 'diff', 'xent', 'dlr']
ORIG_DATA_PATH = f'data/{GAN_NAME}_{DATASET}_recog_png'
ORIG_IMAGES_PATH = osp.join(ORIG_DATA_PATH, 'ims')
ORIG_TENSORS_PATH = osp.join(ORIG_DATA_PATH, 'tensors')
LAT_CODES_PATH = osp.join(ORIG_DATA_PATH, f'{LAT_SPACE}.npy')


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
    elif optim_name == 'RMSProp':
        optim = RMSprop([deltas], lr=lr)

    return optim


def get_dists(embs1, embs2, method='insightface'):
    if method == 'insightface':
        # Use special compute mode for numerical stability
        return torch.cdist(embs1, embs2, 
            compute_mode='donot_use_mm_for_euclid_dist')
    else:
        return 1 - torch.matmul(embs1, embs2.T)
    

def lat2embs(generator, net, lat_codes, transform, few=False, with_tqdm=False,
        return_ims=False):
    all_embs = []
    all_ims = [] if return_ims else None
    # Pad input (if necessary)
    if generator.batch_size % lat_codes.size(0) == 0:
        to_pad = 0
    else:
        to_pad = generator.batch_size - lat_codes.size(0) % generator.batch_size
    
    pad_inp = torch.cat([lat_codes, 
        torch.zeros(to_pad, EMB_SIZE, device=lat_codes.device)], dim=0)

    itr = generator.get_batch_inputs(pad_inp)
    if with_tqdm: itr = tqdm(itr)
    for batch in itr:
        with torch.set_grad_enabled(few):
            # Forward and store
            ims = generator.easy_synthesize(batch.to(DEVICE), **KWARGS)['image']
            if return_ims:
                all_ims.append(ims if few else ims.detach().cpu())
            # Transform, forward and store
            ims = transform(ims)
            ims = ims.unsqueeze(0) if ims.ndim == 3 else ims
            embs = net(ims)
            all_embs.append(embs if few else embs.detach().cpu())
        
    # Extract 'n_orig' samples (since we padded)
    n_orig = lat_codes.size(0)
    if return_ims: all_ims = torch.cat(all_ims)[:n_orig]
        
    return torch.cat(all_embs)[:n_orig], all_ims


def get_curr_preds(generator, net, embs, curr_lats, deltas, transform, device, 
        args):
    # Compute the ORIGINAL images
    orig_embs, orig_ims = lat2embs(generator, net, curr_lats, transform, 
        with_tqdm=True, return_ims=True)

    # Compute the ADVERSARIAL images according to the computed deltas
    adv_lat_cds = curr_lats + deltas
    adv_embs, adv_ims = lat2embs(generator, net, adv_lat_cds, transform, 
        with_tqdm=True, return_ims=True)

    # Get the current predictions according to the distances
    curr_dists = get_dists(adv_embs.to(device), embs, 
        method=args.face_recog_method)
    curr_preds = torch.argmin(curr_dists, 1)
    return adv_embs, adv_ims, curr_preds, orig_embs, orig_ims


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
        nearest_val, _ = torch.max(mod_vals, 1, keepdim=True)
    else: # If they're distances, we are looking for the min
        nearest_val, _ = torch.min(mod_vals, 1, keepdim=True)

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
        xent = F.cross_entropy(input=scores, target=labels, reduction='none')
        # It's the cross-entropy loss, so we want to maximize it
        return -1. * xent.mean()
    # --> 'dlr' loss (difference-of-logits ratio)
    if loss_type == 'dlr':
        assert not use_probs, 'dlr loss works in terms of logits'
        # The numerator
        diff1 = target_val - nearest_val
        # The denominator
        logits = -all_dists
        topk = torch.topk(logits, k=3, dim=1, largest=True, sorted=True)[0]
        diff2 = topk[:, 0] - topk[:, 2] # The first minus the third
        ratio = diff1 / diff2.unsqueeze(1)
        coeff = -1.
        return coeff * ratio.mean()


def init_deltas(random_init, lin_comb, n_vecs, on_surface, ellipse_mat, 
        proj_mat, dirs):
    if random_init:
        if lin_comb:
            ell_mat = torch.diag(ellipse_mat)
            deltas = sample_ellipsoid(ell_mat, n_vecs=n_vecs)
            if on_surface:
                deltas, _ = proj2region(
                    deltas, proj_mat=None, ellipse_mat=ell_mat, check=True, 
                    to_subs=False, dirs=None, on_surface=on_surface
                )
        else:
            # Sample from ellipsoid and project
            deltas = sample_ellipsoid(ellipse_mat, n_vecs=n_vecs)
            deltas, _ = proj2region(deltas, proj_mat, ellipse_mat, check=True, 
                dirs=dirs, on_surface=on_surface)
    else:
        deltas = torch.zeros(n_vecs, EMB_SIZE)

    return deltas.clone().detach()


def get_dists_and_logits(generator, net, codes, transform, orig_embs, 
        frs_method):
    # Make forward pass
    embs, _ = lat2embs(generator, net, codes, transform, few=True)
    # Compute distances
    all_dists = get_dists(embs, orig_embs, method=frs_method)
    # Logits as negative of distances
    logits = -1. * all_dists
    return all_dists, logits


def find_adversaries_autoattack(generator, net, lat_codes, labels, orig_embs, 
        frs_method, transform, lin_comb, attack_type, dirs, red_dirs, dirs_inv,
        red_ellipse_mat, ellipse_mat, ellipse_mat_inv, proj_mat, iters=5, 
        restarts=5, n_target_classes=5):
    # For running AutoAttack, we always think in terms of deltas: find 'deltas'
    # such that g(delta; x) = f(x + delta)  != y. For this we use helper forward 
    # functions and initializations at 0.
    def forward_pass(deltas):
        deltas = deltas.squeeze(3).squeeze(2) # Remove dims for simulating ims
        perturbation = (dirs @ deltas.T).T if lin_comb else deltas
        inp = lat_codes + perturbation
        return get_dists_and_logits(generator, net, inp, transform, orig_embs, 
            frs_method)[1] # The logits
    
    # `eps` is set inside AutoAttack (the passed param should be inocuous)
    adversary = AutoAttack(
        forward_pass, norm='Lsigma2', eps=None, lin_comb=lin_comb, 
        ellipse_mat=ellipse_mat, red_ellipse_mat=red_ellipse_mat, 
        ellipse_mat_inv=ellipse_mat_inv, 
        red_ellipse_mat_inv=None, proj_mat=proj_mat, dirs=dirs,
        dirs_inv=dirs_inv
    )
    adversary.seed = 42
    adversary.attacks_to_run = [attack_type]
    if attack_type == 'fab-t':
        adversary.fab.n_iter = iters # Default is 100
        adversary.fab.n_restarts = restarts # Default is 5
        adversary.fab.n_target_classes = n_target_classes # Default is 9
    elif attack_type == 'fab': # This is INTRACTABLE
        adversary.fab.n_iter = iters
        adversary.fab.n_restarts = restarts
    elif attack_type in ['apgd-ce', 'apgd-dlr', 'apgd-t']:
        # adversary.apgd.n_restarts = 1 # 5
        # adversary.apgd.n_iter = iters
        adversary.apgd_targeted.n_iter = iters
        adversary.apgd_targeted.n_restarts = restarts
        adversary.apgd_targeted.n_target_classes = n_target_classes

    # Here we introduce the zeros, as we will be thinking in terms of deltas!
    deltas_orig = torch.zeros(
        labels.size(0), dirs.size(1) if lin_comb else EMB_SIZE)
    # Simulate images for AutoAttack
    deltas_orig = deltas_orig.unsqueeze(2).unsqueeze(3)
    deltas = adversary.run_standard_evaluation(x_orig=deltas_orig.to(DEVICE), 
        y_orig=labels.to(DEVICE), bs=generator.batch_size)
    # Remove dims for simulating images
    deltas = deltas.squeeze(3).squeeze(2)

    # Compute the predictions
    # Perturb latent codes
    perturbation = (dirs @ deltas.T).T if lin_comb else deltas
    # Compute current predictions and check for attack success
    all_dists, _ = get_dists_and_logits(generator, net, 
        lat_codes+perturbation, transform, orig_embs, frs_method)
    preds = torch.argmin(all_dists, 1)
    success = preds != labels

    check = attack_type not in ['fab', 'fab-t'] # FAB attack is minimum norm
    magnitudes = check_deltas(deltas, lin_comb, red_dirs, red_ellipse_mat, 
        ellipse_mat, proj_mat, check=check)

    return deltas.detach().cpu(), success, magnitudes


def check_deltas(deltas, lin_comb, red_ellipse_mat, ellipse_mat, proj_mat, 
        check=True):
    if lin_comb:
        magnitudes = sq_distance(torch.diag(red_ellipse_mat), 
            deltas.unsqueeze(dim=2))
        if check:
            assert in_ellps(deltas.T, torch.diag(red_ellipse_mat), atol=1e-3)
    else:
        magnitudes = sq_distance(ellipse_mat, deltas.unsqueeze(dim=2))
        if check:
            assert in_subs(deltas.T, proj_mat) and in_ellps(deltas.T, 
                ellipse_mat)

    return magnitudes


def find_adversaries_pgd(generator, net, lat_codes, labels, orig_embs, opt_name, 
        lr, iters, momentum, frs_method, loss_type, transform, ellipse_mat, 
        proj_mat, dirs, dirs_inv, red_ellipse_mat, random_init=True, 
        rand_init_on_surf=True, lin_comb=True, restarts=5):
    ell_mat = red_ellipse_mat if lin_comb else ellipse_mat
    batch_size = lat_codes.size(0)
    # The best deltas (the ones to return at the end)
    best_deltas = torch.zeros(
        batch_size, dirs.size(1) if lin_comb else EMB_SIZE).to(DEVICE)
    best_deltas.requires_grad = False
    for idx_rest in range(restarts):
        # (Re-)Initialize deltas that haven't been successful
        deltas = init_deltas(random_init, lin_comb, batch_size, 
            rand_init_on_surf, ell_mat, proj_mat, dirs)
        deltas = deltas.clone().detach().requires_grad_(True)
        # Their corresponding optimizer
        optim = get_optim(deltas, optim_name=opt_name, lr=lr, momentum=momentum)
        for idx_iter in range(iters):
            # Perturb latent codes
            perturbation = (dirs @ deltas.T).T if lin_comb else deltas
            # Compute current predictions and check for attack success
            all_dists, _ = get_dists_and_logits(generator, net, 
                lat_codes+perturbation, transform, orig_embs, frs_method)
            preds = torch.argmin(all_dists, 1)
            success = preds != labels
            best_deltas[success] = deltas[success].clone().detach()
            if torch.all(success): break
            # Compute loss
            loss = compute_loss(all_dists, labels, loss_type=loss_type,
                use_probs=loss_type!='dlr')
            # Backward and optim step
            optim.zero_grad()
            loss.backward()
            optim.step()
            # Projection
            with torch.no_grad():
                if lin_comb: # Project to low-dimensional ellipsoid
                    # Project this point to inside the ellipse in this space
                    projj, _ = proj2region(
                        deltas, proj_mat=None, ellipse_mat=red_ellipse_mat, 
                        to_subs=False, check=True, on_surface=False,
                        diag_ellipse_mat=True
                    )
                else: # Project to high-dimensional ellipsoid
                    projj, _ = proj2region(best_deltas, proj_mat, ellipse_mat,
                        check=True, dirs=dirs, on_surface=False)
                
                deltas[:] = projj
        
        if torch.all(success): break

    # Final check of deltas
    magnitudes = check_deltas(best_deltas, lin_comb, red_ellipse_mat, 
        ellipse_mat, proj_mat)
    
    return best_deltas.detach().cpu(), success, magnitudes


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


def save_results(results, deltas, successes, magnitudes, num_chunk, args):
    filename = f'results_chunk{num_chunk}of{args.chunks}'
    # Save the deltas and successes
    data_file = osp.join(args.results_dir, f'{filename}.pth')
    if successes.sum() != 0:
        data = { 
            'deltas' : deltas[successes].detach(), 
            'successes' : torch.nonzero(successes).detach(),
            'magnitudes' : magnitudes[successes].detach(), 
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
    avg_magnitude = tot_magnitudes / tot_successes
    info = f'successes:{tot_successes}\n' \
        f'instances:{tot_instances}\n' \
        f'rate:{attack_success_rate:4.2f}\n' \
        f'avg_mag:{avg_magnitude:4.2f}'   

    print_to_log(info, args.final_results)
    args.LOGGER.info(f'Saved all results to {args.final_results}')


def get_all_matrices(attrs2drop=[]):
    assert type(attrs2drop) == list
    # Several matrices:
    # (1) The projection-to-subspace matrix
    # (2) The matrix describing the hyperellipsoid
    # (3) The matrix of directions of our interest
    # (4) The diagonal matrix describing the lower-dimensional hyperellipsoid
    proj_mat, ellipse_mat, dirs, red_ellipse_mat, _ = \
        get_projection_matrices(dataset=DATASET, gan_name=GAN_NAME, 
            attrs2drop=attrs2drop)

    dirs = torch.tensor(dirs, dtype=torch.float32, device=DEVICE)
    proj_mat = torch.tensor(proj_mat, dtype=torch.float32, device=DEVICE)
    ellipse_mat = torch.tensor(ellipse_mat, dtype=torch.float32, device=DEVICE) 
    assert proj_mat.size(0) == proj_mat.size(1) == EMB_SIZE
    assert ellipse_mat.size(0) == ellipse_mat.size(1) == EMB_SIZE

    red_ellipse_mat = torch.tensor(red_ellipse_mat, dtype=torch.float32, 
        device=DEVICE)
    assert red_ellipse_mat.size(0) == dirs.size(1)
    dirs_inv = torch.linalg.pinv(dirs)
    ellipse_mat_inv = torch.linalg.inv(ellipse_mat)
    return (proj_mat, ellipse_mat, ellipse_mat_inv, dirs, dirs_inv, 
        red_ellipse_mat)


def eval_chunk(generator, net, lat_codes, embs, transform, num_chunk, device, 
        args):
    proj_mat, ellipse_mat, ellipse_mat_inv, dirs, dirs_inv, red_ellipse_mat, \
        = get_all_matrices(args.attrs2drop)
    start_time = time()
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
    # Iterate through batches
    deltas, successes, magnitudes, all_labels = [], [], [], []
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
        if args.attack_type == 'manual':
            curr_deltas, succ, mags = find_adversaries_pgd(
                generator, net, btch_cods.to(device), labels, embs, 
                opt_name=args.optim, lr=args.lr, iters=args.iters, 
                momentum=args.momentum, frs_method=args.face_recog_method, 
                loss_type=args.loss, transform=transform, 
                ellipse_mat=ellipse_mat, proj_mat=proj_mat, dirs=dirs, 
                dirs_inv=dirs_inv, red_ellipse_mat=red_ellipse_mat, 
                random_init=True, rand_init_on_surf=not args.not_on_surf, 
                lin_comb=args.lin_comb, restarts=args.restarts
            )
        else:
            curr_deltas, succ, mags = find_adversaries_autoattack(
                generator, net, btch_cods.to(device), labels, embs, 
                frs_method=args.face_recog_method, transform=transform, 
                lin_comb=args.lin_comb, attack_type=args.attack_type, dirs=dirs,
                red_dirs=red_dirs, red_ellipse_mat=red_ellipse_mat, 
                ellipse_mat=ellipse_mat, ellipse_mat_inv=ellipse_mat_inv,
                proj_mat=proj_mat, iters=args.iters, restarts=args.restarts, 
                n_target_classes=args.n_target_classes, dirs_inv=dirs_inv
            )
        
        tot += batch_size
        n_succ += succ.sum()
        # Append results
        successes.append(succ)
        magnitudes.append(mags)
        deltas.append(curr_deltas)
        # Show update
        avg_pert = torch.cat(magnitudes)[torch.cat(successes)].mean()
        pbar.set_description(
            f'-> {n_succ} advs for {tot} IDs -> avg. pert.: {avg_pert:3.4f}')

    deltas = torch.cat(deltas)
    successes = torch.cat(successes)
    magnitudes = torch.cat(magnitudes)
    all_labels = torch.cat(all_labels)
    n_succ = successes.sum()
    tot_time = time() - start_time
    args.LOGGER.info(f'Finished chunk computation. Time={tot_time:3.2f}s')

    # Check the results
    if n_succ == 0: # No successes
        args.LOGGER.info('Didnt find any adversary! =(')
    else:
        # Get the output on the images for which deltas were found
        succ_deltas = deltas[successes]
        succ_mags = magnitudes[successes]
        succ_lat_codes = lat_cods_chunk[successes]
        if args.lin_comb:
            succ_deltas = (dirs.to(deltas.device) @ succ_deltas.T).T

        adv_embs, adv_ims, curr_preds, _, orig_ims = get_curr_preds(
            generator, net, embs, succ_lat_codes,
            succ_deltas, transform, device, args
        )
        # Check they are indeed adversarial
        assert check_advs(all_labels[successes], curr_preds, successes, args)
        # Compute images of the IDs with which people are being confused
        lat_cods_conf = lat_codes[curr_preds]
        _, conf_adv_ims, _, _, conf_ims = get_curr_preds(
            generator, net, embs, lat_cods_conf, 
            torch.zeros_like(lat_cods_conf), transform, device, args
        )
        assert torch.equal(conf_adv_ims, conf_ims)
        # Plot the images and their adversaries
        plot_advs(orig_ims, all_labels[successes], adv_ims, curr_preds, 
            conf_ims, args, succ_mags)
        avg_pert = magnitudes[successes].mean().item()
        args.LOGGER.info(f'-> Found {n_succ} advs for {tot} IDs ' \
            f'-> avg. pert.: {avg_pert:3.4f}')
    
    # Log the results
    results = {
        'successes' : n_succ,
        'instances' : len(all_labels),
        'avg_mags' : avg_pert if n_succ != 0 else 0
    }
    # Store adversaries and the successes
    log_file = save_results(results, deltas, successes, magnitudes, num_chunk, 
        args)
    return log_file

    
def plot_advs(orig_ims, orig_labels, adv_ims, adv_labels, confu, args, mags):
    # Show in a plot
    args.LOGGER.info(f'Plotting adversaries')
    for ori_im, ori_lab, adv_im, adv_lab, conf, mag in zip(orig_ims, 
            orig_labels, adv_ims, adv_labels, confu, mags):
        plt.figure()
        plt.subplot(131); plt.imshow(ori_im.cpu().permute(1, 2, 0).numpy())
        plt.axis('off'); plt.title('Original')

        plt.subplot(132); plt.imshow(adv_im.cpu().permute(1, 2, 0).numpy())
        plt.axis('off'); plt.title(f'Adversary ({mag:4.3f})')

        plt.subplot(133); plt.imshow(conf.cpu().permute(1, 2, 0).numpy())
        plt.axis('off'); plt.title('Prediction')

        plt.tight_layout()
        path = osp.join(args.figs_dir, f'ori_{ori_lab}_adv_{adv_lab}.jpg')
        plt.savefig(path, bbox_inches='tight', dpi=400)
        plt.close()
    
