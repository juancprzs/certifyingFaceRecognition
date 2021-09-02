import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Torch imports
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
# Face recognition models
from models.iresnet import iresnet50
from facenet_pytorch import InceptionResnetV1
# Local imports
from utils.logger import setup_logger
from models.mod_stylegan_generator import ModStyleGANGenerator
from proj_utils import (get_projection_matrices, sample_ellipsoid,
    project_to_region_pytorch, set_seed)
# To handle too many open files
torch.multiprocessing.set_sharing_strategy('file_system')

assert torch.cuda.is_available()

# For deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True

OPTIMS = ['Adam', 'SGD']
FRS_METHODS = ['insightface', 'facenet']
LOSS_TYPES = ['away', 'nearest', 'diff', 'xent']
INP_RESOLUTIONS = { 'insightface' : 112, 'facenet' : 160 }

# # Script constants
# Misc
STD = 0.5
MEAN = 0.5
PLOT = False
EMB_SIZE = 512
LAT_SPACE = 'w'
BATCH_SIZE = 32
N_SHOW_ERRS = 5
DATASET = 'ffhq'
IMAGE_EXT = 'png'
GAN_NAME = 'stylegan'
DEVICE = torch.device('cuda')
# Paths
ORIG_DATA_PATH = f'data/{GAN_NAME}_{DATASET}_recog_png'
ORIG_IMAGES_PATH = osp.join(ORIG_DATA_PATH, 'ims')
ORIG_TENSORS_PATH = osp.join(ORIG_DATA_PATH, 'tensors')
WEIGHTS_PATH = 'weights/ms1mv3_arcface_r50/backbone.pth'
DATA_PATH = f'results/{GAN_NAME}_{DATASET}_recog_png_perts/data'

def get_transform():
    def resize(x):
        x = x.unsqueeze(0) if x.ndim != 4 else x
        interp = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', 
            align_corners=False)
        return interp.squeeze(0)

    normalize = Normalize((MEAN, MEAN, MEAN), (STD, STD, STD))
    return Compose([resize, normalize])
# Transforms
TRANSFORM = get_transform()

PROJ_MAT, ELLIPSE_MAT, DIRS, _ = get_projection_matrices(dataset=DATASET, 
    gan_name=GAN_NAME)
DIRS = torch.tensor(DIRS, dtype=torch.float32, device=DEVICE)
PROJ_MAT = torch.tensor(PROJ_MAT, dtype=torch.float32, device=DEVICE)
ELLIPSE_MAT = torch.tensor(ELLIPSE_MAT, dtype=torch.float32, device=DEVICE) 
assert PROJ_MAT.shape[0] == PROJ_MAT.shape[1] == EMB_SIZE
assert ELLIPSE_MAT.shape[0] == ELLIPSE_MAT.shape[1] == EMB_SIZE
set_seed(DEVICE, seed=2)


# # Input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Compute semantic adversaries')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum for SGD')
    parser.add_argument('--loss', type=str, default='xent', choices=LOSS_TYPES,
                        help='Loss to optimize')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    parser.add_argument('--face-recog-method', type=str, default='insightface', 
                        choices=FRS_METHODS, help='Face recognition system to use')
    parser.add_argument('--optim', type=str, default='SGD', choices=OPTIMS,
                        help='Optimizer to use')
    parser.add_argument('--iters', type=int, default=10, 
                        help='Optimization iterations per instance')
    parser.add_argument('--not-on-surf', action='store_true', default=False,
                        help='Random initialization is NOT on region surface')
    return parser.parse_args()


def args2text(args):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    return text


args = parse_args()


# Script argument-dependent constants
IMG_SIZE = INP_RESOLUTIONS[args.face_recog_method]

# The model and the latent codes
LOGGER = setup_logger(args.output_dir, logger_name='generate_data')
LOGGER.info(args2text(args))
LOGGER.info(f'Initializing generator.')
GENERATOR = ModStyleGANGenerator('stylegan_ffhq', LOGGER)
GENERATOR.model.eval()
KWARGS = { 'latent_space_type' : LAT_SPACE }
LAT_CODES_PATH = osp.join(ORIG_DATA_PATH, f'{LAT_SPACE}.npy')
LAT_CODES = torch.from_numpy(
    GENERATOR.preprocess(np.load(LAT_CODES_PATH), **KWARGS)
)


class SimpleImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.im_names = [osp.basename(x) 
            for x in glob(osp.join(root, f'*.{IMAGE_EXT}'))]
        self.im_names = sorted(self.im_names, 
            key=lambda x: int(x.replace(f'.{IMAGE_EXT}', '')))

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        img_name = osp.join(self.root, osp.basename(self.im_names[idx]))
        image = pil_loader(img_name)
        if self.transform: image = self.transform(image)

        return image, osp.basename(img_name)


class SimpleTensorDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.tensor_names = [osp.basename(x) 
            for x in glob(osp.join(root, f'*.pth'))]
        self.tensor_names = sorted(self.tensor_names, 
            key=lambda x: int(x.replace(f'.pth', '')))

    def __len__(self):
        return len(self.tensor_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        tensor_name = osp.join(self.root, osp.basename(self.tensor_names[idx]))
        image = torch.load(tensor_name, map_location=DEVICE)
        if self.transform: image = self.transform(image)

        return image, osp.basename(tensor_name)


class ImageFolderWithPaths(ImageFolder):
    # Taken from
    # https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # From YY/.../XX/data/class_000/000.png to class_000/000.png
        path = osp.join(*path.split('/')[-2:])
        # Return a new tuple
        return (original_tuple[0], path)


def pil_loader(path: str) -> Image.Image:
    # Taken from 
    # https://pytorch.org/vision/0.8/_modules/torchvision/datasets/folder.html#ImageFolder
    # open path as file to avoid ResourceWarning 
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_net(method='insightface'):
    if method == 'insightface':
        net = iresnet50(False, fp16=False) # pretrained==False
        net.load_state_dict(torch.load(WEIGHTS_PATH))
    else:
        net = InceptionResnetV1(pretrained='casia-webface')
    
    return net.to(DEVICE).eval()


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


def get_embs(net, dataloader, original=True, to_cpu=True):
    
    def check_ims(names):
        try:
            curr_nums = [int(x.replace(f'.{IMAGE_EXT}', '')) for x in names]
        except:
            curr_nums = [int(x.replace(f'.pth', '')) for x in names]
        expected_curr_num = range(idx*BATCH_SIZE, 
            idx*BATCH_SIZE+len(curr_nums))
        assert curr_nums == list(expected_curr_num), \
            'Something wrong with loading the original images'
    
    embs, others = [], []
    for idx, (imgs, other) in enumerate(tqdm(dataloader)):
        if original: # The dataset returns image names, i.e. names = other
            # Check everything is ok
            check_ims(other)

        others.extend(other)
        # Get embeddings and store
        out = net(imgs.to(DEVICE))
        if to_cpu:
            embs.append(out.detach().cpu())
        else:
            embs.append(out)
        
    
    embs = torch.cat(embs, dim=0)
    assert (len(embs.shape) == 2) and (embs.shape[1] == EMB_SIZE) and \
        (embs.shape[0] == len(dataloader.dataset))
    
    return embs, others


def compute_embs(net, original=False, dataset=None, with_grad=False):
    if dataset is None:
        if original:
            print('Computing ORIGINAL embeddings from images')
            dataset = SimpleTensorDataset(root=ORIG_TENSORS_PATH, 
                transform=TRANSFORM)
        else:
            print('Computing NEW embeddings from images')
            dataset = ImageFolderWithPaths(root=DATA_PATH, transform=TRANSFORM)
            # ImageFolder will have, possibly, screwed up the mapping from class 
            # name to idx. Like: dataset.class_to_idx can be
            # {'class_552': 4552, 'class_553': 4553, 'class_554': 4554} instead 
            # of the obvious. We correct for this here:
            dataset.class_to_idx = { k : int(k.split('_')[1]) 
                for k in dataset.class_to_idx.keys() }
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0)
    
    with torch.set_grad_enabled(with_grad):
        embs, others = get_embs(net, dataloader, original=original, 
            to_cpu=not with_grad)

    return embs, others


def lat2embs(net, lat_codes, few=False, with_tqdm=False):
    all_ims, all_embs = [], []
    itr = GENERATOR.get_batch_inputs(lat_codes)
    if with_tqdm: itr = tqdm(itr)
    for batch in itr:
        with torch.set_grad_enabled(few):
            # Forward and store
            ims = GENERATOR.easy_synthesize(batch.to(DEVICE), **KWARGS)['image']
            all_ims.append(ims if few else ims.detach().cpu())
            # Transform, forward and store
            ims = TRANSFORM(ims)
            ims = ims.unsqueeze(0) if ims.ndim == 3 else ims
            embs = net(ims)
            all_embs.append(embs if few else embs.detach().cpu())

    return torch.cat(all_embs), torch.cat(all_ims)


def get_pairwise_dists(embs1, embs2, method='insightface'):
    if method == 'insightface':
        diff = embs1 - embs2
        return torch.norm(diff, dim=1)
    else:
        dot = (embs1 * embs2).sum(dim=1)
        return 1 - dot


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
        momentum, frs_method, loss_type, random_init=True, 
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
        embs, _ = lat2embs(net, lat_codes + deltas, few=True)
        # Compute current predictions and check for attack success
        all_dists = get_dists(embs, orig_embs, method=frs_method)
        preds = torch.argmin(all_dists, 1)
        success = preds != labels
        if torch.all(success):
            break
        # Compute loss
        loss = compute_loss(all_dists, labels, loss_type=loss_type)
        # Backward and optim step
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Projection
        ''''
        with torch.no_grad():
            proj, _ = project_to_region_pytorch(deltas, PROJ_MAT, ELLIPSE_MAT, 
                check=True, dirs=DIRS, on_surface=False)
            # Modify deltas for which attack has not been successful yet
            deltas[~success] = proj[~success]
        '''

    return deltas.detach().cpu(), success


def main():
    # DNN
    net = get_net(method=args.face_recog_method)
    LOGGER.info(f'Generating original images and embeddings')
    embs, all_ims = lat2embs(net, LAT_CODES, with_tqdm=True)
    embs = embs.to(DEVICE)
    # --------------------------------------------------------------------------
    LOGGER.info(f'Computing adversaries')
    n_succ, tot = 0, 0
    deltas, successes = [], []
    pbar = tqdm(GENERATOR.get_batch_inputs(LAT_CODES))
    for idx, btch_cods in enumerate(pbar):
        batch_size = btch_cods.size(0)
        labels = torch.arange(idx*batch_size, (idx+1)*batch_size, device=DEVICE)
        curr_deltas, succ = find_adversaries(
            net, btch_cods.to(DEVICE), labels, embs, optim_name=args.optim, 
            lr=args.lr, iters=args.iters, momentum=args.momentum, 
            frs_method=args.face_recog_method, loss_type=args.loss, 
            random_init=True, rand_init_on_surf=not args.not_on_surf
        )
        tot += batch_size
        n_succ += succ.sum()
        pbar.set_description(f'-> {n_succ} adversaries for {tot} identities')
        successes.append(succ)
        deltas.append(curr_deltas)

    deltas, successes = torch.cat(deltas), torch.cat(successes)
    n_succ = successes.sum()
    if n_succ == 0:
        LOGGER.info(f'Didnt find any adversary! :(')
    else:
        LOGGER.info(f'Total: {n_succ} adversaries for {embs.size(0)} identities')
        # Compute the adversarial images according to the computed deltas
        adv_lat_codes = LAT_CODES[successes] + deltas[successes]
        # Pad input (if necessary)
        to_pad = GENERATOR.batch_size - adv_lat_codes.size(0) % GENERATOR.batch_size
        pad_inp = torch.cat([adv_lat_codes, torch.zeros(to_pad, EMB_SIZE)], dim=0)
        adv_embs, adv_ims = lat2embs(net, pad_inp, with_tqdm=True)
        # Extract (since we padded)
        adv_embs, adv_ims = adv_embs[:n_succ], adv_ims[:n_succ]

        # Check where the predictions differ from the labels
        curr_dists = get_dists(adv_embs.to(DEVICE), embs, 
            method=args.face_recog_method)
        expctd_labels = torch.nonzero(successes).squeeze()
        curr_labels = torch.argmin(curr_dists, 1)
        where_adv = expctd_labels != curr_labels
        assert torch.all(where_adv), 'All these instances should be adversaries'

        # Extract the original images and the identities with 
        # which the people are being confused
        orig, confu = all_ims[successes], all_ims[curr_labels]

        # Show in a plot
        LOGGER.info(f'Plotting adversaries')
        for idx, (ori, adv, conf) in enumerate(zip(orig, adv_ims, confu)):
            plt.figure()
            plt.subplot(131); plt.imshow(ori.cpu().permute(1, 2, 0).numpy())
            plt.axis('off'); plt.title('Original')

            plt.subplot(132); plt.imshow(adv.cpu().permute(1, 2, 0).numpy())
            plt.axis('off'); plt.title('Adversary')

            plt.subplot(133); plt.imshow(conf.cpu().permute(1, 2, 0).numpy())
            plt.axis('off'); plt.title('Prediction')

            plt.tight_layout()
            path = osp.join(args.output_dir, f'id_{idx}.png')
            plt.savefig(path, bbox_inches='tight', dpi=400)
            plt.close()

        
    if PLOT:
        # ------------------------------------------------------------------------------
        # # # # # Embeddings of the original images
        embs, im_names = compute_embs(net, original=True)
        orig_labels = torch.tensor([int(x.replace(f'.pth','')) for x in im_names])
        orig_dists = get_dists(embs, embs, method=args.face_recog_method)
        # Get inter-cluster distances (i.e. between clusters)
        orig_dists_copy = orig_dists.numpy().copy()
        orig_dists_copy[np.eye(orig_dists_copy.shape[0], dtype=bool)] = float('inf')
        min_vals = orig_dists_copy.min(axis=0) # Minimum w/o considering diagonal

        if PLOT:
            plt.hist(min_vals, edgecolor='black', linewidth=1, bins=20)
            mean, std = min_vals.mean(), min_vals.std()
            minn, maxx = min_vals.min(), min_vals.max()
            titl = 'Dists to closest cluster\n' \
                f'mean={mean:3.2f}, std={std:3.2f}, max={maxx:3.2f}, min={minn:3.2f}'
            plt.title(titl, fontsize=24)
            plt.show()

        # Embeddings of the new images
        new_embs, new_im_names = compute_embs(net, original=False)
        targets = torch.tensor([int(x.split('/')[0].split('_')[1]) 
            for x in new_im_names])


        # Perturbations per ID
        perts_per_id = new_embs.size(0) // embs.size(0)

        # Compute distances within clusters
        dists = get_dists(new_embs, embs, method=args.face_recog_method)
        dists_to_centroid = torch.gather(dists, 1, targets.view(-1, 1))
        if PLOT:
            plt.hist(dists_to_centroid.numpy(), edgecolor='black', linewidth=1, bins=20)
            mean, std = dists_to_centroid.mean(), dists_to_centroid.std()
            minn, maxx = dists_to_centroid.min(), dists_to_centroid.max()
            titl = 'Dists to correct centroid\n' \
                f'mean={mean:3.2f}, std={std:3.2f}, max={maxx:3.2f}, min={minn:3.2f}'
            plt.title(titl, fontsize=24)
            plt.show()

        # # # # # Check the accuracy of the network on these images
        # Nearest neighbor
        argsort = torch.argsort(dists, dim=1, descending=False) # Ascending order
        assigns = argsort[:, 0] # Equivalent to torch.argmin(dists, dim=1)
        # Prediction is: each instance is of the same class as the instance at embs[assigns]
        preds = orig_labels[assigns]
        correct = (preds == targets).sum().item()
        n_wrong = new_embs.size(0) - correct
        acc = correct / new_embs.size(0)
        print(f'Accuracy is {100.*acc:3.2f}%')


        # Create DataFrame for easier extraction of data
        dists_to_preds = torch.gather(dists, 1, preds.view(-1, 1))
        df = pd.DataFrame({
            'im_path' : new_im_names, 
            'target' : targets, 
            'dist2target' : dists_to_centroid.squeeze(), 
            'pred' : assigns, 
            'dist2pred' : dists_to_preds.squeeze()
        })


        where_wrong = preds != targets
        # Get the counts through hist and then close the figure
        counts, _, _ = plt.hist(targets[where_wrong].numpy(), 
            range=(0, embs.size(0)-1), bins=embs.size(0))
        plt.close()
        if PLOT:
            counts, _, _ = plt.hist(targets[where_wrong].numpy(), 
                range=(0, embs.size(0)-1), bins=embs.size(0))
            plt.title(f'Errors per ID | Total errors: {n_wrong}', fontsize=24)
            plt.xlabel('ID', fontsize=20)
            plt.ylabel('Counts', fontsize=20)
            plt.show()

            bins = np.linspace(-0.5, perts_per_id+0.5, perts_per_id+2)
            new_counts, _, _ = plt.hist(counts, bins=bins, edgecolor='black', 
                linewidth=1)
            new_n_wrong = int(np.dot(new_counts, np.arange(perts_per_id+1)))
            assert n_wrong == new_n_wrong
            plt.title('Distrib of # of errors', fontsize=24)
            plt.xlabel('# of errors', fontsize=20)
            plt.ylabel('Counts', fontsize=20)
            plt.show()

            # Cumulative distribution of errors
            sorted_errs = np.sort(counts)[::-1]
            cumsum = np.cumsum(sorted_errs)
            cumsum = cumsum[cumsum<new_n_wrong]
            cum_distrib = np.concatenate([cumsum, np.array(n_wrong).reshape(1,)])
            plt.bar(x=range(1, len(cum_distrib)+1), height=cum_distrib)
            plt.title(
                f'Cumulative distribution of errors | {n_wrong} errors from '\
                f'{len(cum_distrib)} identities', 
                fontsize=24
            )
            plt.xlabel('Sorted IDs', fontsize=20)
            plt.ylabel('Total counts', fontsize=20)
            plt.show()


        # Troubling identities
        sorted_ids = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_ids]

        troubling_ids = sorted_ids[sorted_counts > 0]
        troubling_counts = sorted_counts[sorted_counts > 0]


        # Plot some troubling identities
        if PLOT:
            print(f'Will show {N_SHOW_ERRS} identities with multiple errors')
            for tr_id in troubling_ids[:N_SHOW_ERRS]:
                fig, axes = plt.subplots(nrows=4, ncols=perts_per_id)
                # The anchor image for this ID
                orig_im_path = osp.join(
                    ORIG_IMAGES_PATH, f'{tr_id}'.zfill(6) + f'.{IMAGE_EXT}')
                orig_im = mpimg.imread(orig_im_path)
                mid_axes = axes[0, perts_per_id//2]
                mid_axes.imshow(orig_im); mid_axes.set_title(f'{tr_id}', fontsize=20)

                # The images whose target was this ID
                where_bool = df['im_path'].apply(
                    lambda x: x.startswith('class_' + f'{tr_id}'.zfill(3) + '/'))
                these_ims_df = df.loc[where_bool]
                these_ims_df.sort_values(by='im_path')

                # Each image
                for index, (_, row) in enumerate(these_ims_df.iterrows()):
                    dist2target = row['dist2target']
                    im_name = row['im_path']
                    titl = f'{im_name}\nDist2target = {dist2target:.2f}'
                    if row['target'] == row['pred']: # Correctly classified
                        axis_idx = 1
                    else: # Incorrectly classified
                        axis_idx = 2
                        dist2pred = row['dist2pred']
                        titl += f'\nDist2pred = {dist2pred:.2f}'
                        # Load the anchor image
                        this_pred = row['pred']
                        anchor_im_path = osp.join(ORIG_IMAGES_PATH, 
                            f'{this_pred}'.zfill(6) + f'.{IMAGE_EXT}')
                        anchor_im = mpimg.imread(anchor_im_path)
                        axes[3, index].imshow(anchor_im)
                        axes[3, index].set_title(f'{this_pred}', fontsize=20)

                    # The image itself
                    this_im = mpimg.imread(osp.join(DATA_PATH, im_name))
                    axes[axis_idx, index].imshow(this_im)
                    axes[axis_idx, index].set_title(titl, fontsize=20)

                # Turn off all axes
                for idx in range(4):
                    for jdx in range(perts_per_id): 
                        axes[idx, jdx].axis('off')
                
                plt.show()


if __name__ == '__main__':
    main()
