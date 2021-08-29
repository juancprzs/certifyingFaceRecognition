import cv2
import torch
import random
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Torch imports
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
    project_to_region_pytorch)
# To handle too many open files
torch.multiprocessing.set_sharing_strategy('file_system')

# For deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)


def set_seed(device, seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


# Constants
STD = 0.5
MEAN = 0.5
PLOT = False
EMB_SIZE = 512
LAT_SPACE = 'w'
BATCH_SIZE = 32
N_SHOW_ERRS = 5
DATASET = 'ffhq'
GAN_NAME = 'stylegan'
METHOD = 'insightface'
OUTPUT_DIR = 'remove_soon'
FRS_METHODS = ['insightface', 'facenet']
IMG_SIZE = 112 if METHOD == 'insightface' else 160
WEIGHTS_PATH = 'weights/ms1mv3_arcface_r50/backbone.pth'
ORIG_IMAGES_PATH = f'data/{GAN_NAME}_{DATASET}_recog_debug'
DATA_PATH = f'results/{GAN_NAME}_{DATASET}_recog_debug_perts/data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORM = Compose([
    Resize(size=(IMG_SIZE, IMG_SIZE), 
        interpolation=InterpolationMode.BILINEAR), # cv2's default
    ToTensor(),
    Normalize((MEAN, MEAN, MEAN), (STD, STD, STD))
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
MODEL = ModStyleGANGenerator('stylegan_ffhq', LOGGER)
KWARGS = { 'latent_space_type' : LAT_SPACE }
LAT_CODES_PATH = osp.join(ORIG_IMAGES_PATH, f'{LAT_SPACE}.npy')
LAT_CODES = torch.from_numpy(
    MODEL.preprocess(np.load(LAT_CODES_PATH), **KWARGS)
)


class SimpleDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.im_names = [osp.basename(x) for x in glob(osp.join(root, '*.jpg'))]
        self.im_names = sorted(self.im_names, 
            key=lambda x: int(x.replace('.jpg', '')))

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        img_name = osp.join(self.root, osp.basename(self.im_names[idx]))
        image = pil_loader(img_name)
        if self.transform: image = self.transform(image)

        return image, osp.basename(img_name)


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
        # From YY/.../XX/data/class_000/000.jpg to class_000/000.jpg
        path = osp.join(*path.split('/')[-2:])
        # Return a new tuple
        return (original_tuple[0], path)


def get_images_from_latent(lat_codes):
    outputs = []
    for idx, batch in enumerate(MODEL.get_batch_inputs(lat_codes)):
        out = MODEL.easy_synthesize(batch, **KWARGS)
        for image in out['image']:
            outputs.append(image)
        
    return torch.cat([x.unsqueeze(0) for x in outputs])


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


def get_optim(deltas, optim_name='SGD',lr=0.001, momentum=0.9):
    if optim_name == 'SGD':
        optim = SGD([deltas], lr=lr, momentum=0.9)
    elif optim_name == 'Adam':
        optim = Adam([deltas], lr=lr)

    return optim


def get_dists(embs1, embs2, method='insightface'):
    if method == 'insightface':
        return torch.cdist(embs1, embs2)
    else:
        return 1 - torch.matmul(embs1, embs2.T)


def get_embs(net, dataloader, original=True, to_cpu=True):
    
    def check_ims(names):
        curr_nums = [int(x.replace('.jpg', '')) for x in names]
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
        # if idx == 10: break # REMOVE THIS!!
    
    embs = torch.cat(embs, dim=0)
    assert (len(embs.shape) == 2) and (embs.shape[1] == EMB_SIZE) and \
        # (embs.shape[0] == len(dataloader.dataset)) # REMOVE THIS!!
    
    return embs, others


def compute_embs(net, original=False, dataset=None, with_grad=False):
    if dataset is None:
        if original:
            print('Computing ORIGINAL embeddings')
            dataset = SimpleDataset(root=ORIG_IMAGES_PATH, transform=TRANSFORM)
        else:
            print('Computing NEW embeddings')
            dataset = ImageFolderWithPaths(root=DATA_PATH, transform=TRANSFORM)
            # ImageFolder will have, possibly, screwed up the mapping from class 
            # name to idx. Like: dataset.class_to_idx can be
            # {'class_552': 4552, 'class_553': 4553, 'class_554': 4554} instead 
            # of the obvious. We correct for this here:
            dataset.class_to_idx = { k : int(k.split('_')[1]) 
                for k in dataset.class_to_idx.keys() }
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=1)
    if with_grad:
        embs = get_embs(net, dataloader, original=original, to_cpu=False)
    else:
        with torch.no_grad():
            embs = get_embs(net, dataloader, original=original, to_cpu=True)

    return embs



def lat2embs(net, lat_codes, few=False):
    ims = get_images_from_latent(lat_codes)
    if not few:
        ims = ims.cpu()
    
    # Resize images
    ims = F.interpolate(ims, size=(IMG_SIZE, IMG_SIZE), mode='bilinear')

    # Normalize images
    num_dims = len(ims.shape)
    std = torch.tensor(STD).reshape(num_dims * (1,)).to(ims.device)
    mean = torch.tensor(MEAN).reshape(num_dims * (1,)).to(ims.device)

    # Forward through network
    local_dataset = TensorDataset(ims, torch.empty(ims.size(0))) # 2arg=whatever
    embs, _ = compute_embs(net, original=False, dataset=local_dataset, 
        with_grad=few)
    return embs


def find_adversaries(net, lat_codes, orig_embs, orig_labels, iters=10, 
        random_init=False, on_surface=True):
    # Initialize deltas
    if random_init:
        # Sample from ellipsoid and project
        deltas = sample_ellipsoid(ELLIPSE_MAT, n_vecs=lat_codes.size(0))
        deltas, _ = project_to_region_pytorch(deltas, PROJ_MAT, 
            ELLIPSE_MAT, check=True, dirs=DIRS, on_surface=on_surface)
        # Transform to tensor
        deltas = deltas.clone().detach().requires_grad_(True)
    else:
        deltas = torch.zeros_like(lat_codes, requires_grad=True)
    
    # Their corresponding optimizer
    optim = get_optim(deltas, optim_name='Adam')
    for idx_iter in range(iters):
        print(f'Opt. step #{idx_iter}. deltas.abs().sum()={deltas.abs().sum()}')
        embs = lat2embs(net, lat_codes + deltas, few=True)
        import pdb; pdb.set_trace()
        loss = embs.mean()
        # Backward and optim step
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Projection
        proj, _ = project_to_region_pytorch(deltas, PROJ_MAT, ELLIPSE_MAT, 
            check=True, dirs=DIRS, on_surface=False)
        deltas = proj.detach()


    return deltas


# DNN
net = get_net(method=METHOD)
# # # # # Embeddings of the original images
embs, im_names = compute_embs(net, original=True)
orig_labels = torch.tensor([int(x.replace('.jpg','')) for x in im_names])
orig_dists = get_dists(embs, embs, method=METHOD)
# Get inter-cluster distances (i.e. between clusters)
orig_dists_copy = orig_dists.numpy().copy()
orig_dists_copy[np.eye(orig_dists_copy.shape[0], dtype=bool)] = +float('inf')
min_vals = orig_dists_copy.min(axis=0) # Minimum without considering diagonal

# ------------------------------------------------------------------------------
NUM = 5
some_codes = LAT_CODES[:NUM]
some_codes.requires_grad = True
deltas = find_adversaries(net, some_codes.to(DEVICE), embs, orig_labels, 
    random_init=False)

loss = local_embs.mean()
loss.backward()

print('lat codes grad: ', some_codes.grad)
# ------------------------------------------------------------------------------

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
        orig_im_path = osp.join(ORIG_IMAGES_PATH, f'{tr_id}'.zfill(6) + '.jpg')
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
                    f'{this_pred}'.zfill(6) + '.jpg')
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
