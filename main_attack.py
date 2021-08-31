import cv2
import torch
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

# For deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)
assert torch.cuda.is_available()

# Constants
STD = 0.5
MEAN = 0.5
PLOT = False
EMB_SIZE = 512
LAT_SPACE = 'w'
BATCH_SIZE = 32
N_SHOW_ERRS = 5
IMAGE_EXT = 'png'
DATASET = 'ffhq'
GAN_NAME = 'stylegan'
METHOD = 'insightface'
OUTPUT_DIR = 'remove_soon'
DEVICE = torch.device('cuda')
FRS_METHODS = ['insightface', 'facenet']
IMG_SIZE = 112 if METHOD == 'insightface' else 160

# Paths
ORIG_DATA_PATH = f'data/{GAN_NAME}_{DATASET}_recog_png'
ORIG_IMAGES_PATH = osp.join(ORIG_DATA_PATH, 'ims')
ORIG_TENSORS_PATH = osp.join(ORIG_DATA_PATH, 'tensors')
WEIGHTS_PATH = 'weights/ms1mv3_arcface_r50/backbone.pth'
DATA_PATH = f'results/{GAN_NAME}_{DATASET}_recog_png_perts/data'

# Transforms
def resize(x):
    x = x.unsqueeze(0) if x.ndim != 4 else x
    interp = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', 
        align_corners=False)
    return interp.squeeze(0)

normalize = Normalize((MEAN, MEAN, MEAN), (STD, STD, STD))

TRANSFORM = Compose([
    resize,
    normalize
])

PROJ_MAT, ELLIPSE_MAT, DIRS, _ = get_projection_matrices(dataset=DATASET, 
    gan_name=GAN_NAME)
DIRS = torch.tensor(DIRS, dtype=torch.float32, device=DEVICE)
PROJ_MAT = torch.tensor(PROJ_MAT, dtype=torch.float32, device=DEVICE)
ELLIPSE_MAT = torch.tensor(ELLIPSE_MAT, dtype=torch.float32, device=DEVICE) 
assert METHOD in FRS_METHODS
assert PROJ_MAT.shape[0] == PROJ_MAT.shape[1] == EMB_SIZE
assert ELLIPSE_MAT.shape[0] == ELLIPSE_MAT.shape[1] == EMB_SIZE

set_seed(DEVICE, seed=2)
# The model and the latent codes
LOGGER = setup_logger(OUTPUT_DIR, logger_name='generate_data')
LOGGER.info(f'Initializing generator.')
GENERATOR = ModStyleGANGenerator('stylegan_ffhq', LOGGER)
GENERATOR.model.eval()
for child in GENERATOR.model.children():
    if type(child) == nn.BatchNorm2d:
        child.track_running_stats = False

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
        optim = SGD([deltas], lr=lr, momentum=0.9)
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


def find_adversaries(net, lat_codes, labels, orig_embs, iters=100, 
        random_init=True, rand_init_on_surf=True):
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
    optim = get_optim(deltas, optim_name='Adam')
    for idx_iter in range(iters):
        print(f'Opt. step #{idx_iter}. deltas.abs().sum()={deltas.abs().sum()}')
        # Get embeddings from perturbed latent codes
        embs, _ = lat2embs(net, lat_codes + deltas, few=True)
        import pdb; pdb.set_trace()
        # Compute current predictions and check for attack success
        all_dists = get_dists(embs, orig_embs, method=METHOD)
        preds = torch.argmin(all_dists, dim=1)
        success = preds != labels
        targets = orig_embs[labels]
        dists = get_pairwise_dists(embs, targets, method=METHOD)
        loss = -dists.mean()
        print('loss:', loss.item())
        # Backward and optim step
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Projection
        proj, _ = project_to_region_pytorch(deltas, PROJ_MAT, ELLIPSE_MAT, 
            check=True, dirs=DIRS, on_surface=False)
        # Modify deltas for which attack has not been successful yet
        deltas[~success] = proj[~success].detach()

    return deltas


def main():
    # DNN
    net = get_net(method=METHOD)
    print('Generating original images and embeddings')
    all_embs, all_ims = lat2embs(net, LAT_CODES[:100], with_tqdm=True)
    all_embs = all_embs.to(DEVICE)
    # --------------------------------------------------------------------------
    print('Computing adversaries')
    for idx, btch_cods in enumerate(GENERATOR.get_batch_inputs(LAT_CODES)):
        batch_size = btch_cods.size(0)
        labels = torch.arange(idx*batch_size, (idx+1)*batch_size, device=DEVICE)
        deltas = find_adversaries(net, btch_cods.to(DEVICE), labels, 
            all_embs)
        import pdb; pdb.set_trace()
        
    temp_embs, temp_ims = lat2embs(net, LAT_CODES[:NUM], few=True)
    (temp_ims - all_ims[:NUM].cuda()).sum((1,2,3))
    orig_labels = 0
    some_codes.requires_grad = True
    deltas = find_adversaries(net, LAT_CODES[NUM:2*NUM].to(DEVICE), embs, orig_labels, 
        random_init=False)
    deltas = find_adversaries(net, LAT_CODES[2*NUM:3*NUM].to(DEVICE), embs, orig_labels, 
        random_init=False)


    loss = local_embs.mean()
    loss.backward()

    print('lat codes grad: ', some_codes.grad)
    # ------------------------------------------------------------------------------
    # # # # # Embeddings of the original images
    embs, im_names = compute_embs(net, original=True)
    orig_labels = torch.tensor([int(x.replace(f'.pth','')) for x in im_names])
    orig_dists = get_dists(embs, embs, method=METHOD)
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
    dists = get_dists(new_embs, embs, method=METHOD)
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


    import pdb; pdb.set_trace()
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5_000, random_state=0).fit(embs.detach().numpy())
    print(kmeans.labels_)
    print(label.numpy())

if __name__ == '__main__':
    main()
