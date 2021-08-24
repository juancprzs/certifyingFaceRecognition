import cv2
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import OrderedDict
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
# Face recognition models
from models.iresnet import iresnet50
from facenet_pytorch import InceptionResnetV1
# To handle too many open files
torch.multiprocessing.set_sharing_strategy('file_system')

PLOT = False
EMB_SIZE = 512
BATCH_SIZE = 32
METHOD = 'insightface'
FRS_METHODS = ['insightface', 'facenet']
IMG_SIZE = 112 if METHOD == 'insightface' else 160
ORIG_IMAGES_PATH = 'data/stylegan_ffhq_recog_debug'
WEIGHTS_PATH = 'weights/ms1mv3_arcface_r50/backbone.pth'
DATA_PATH = 'results/stylegan_ffhq_recog_debug_perts/data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORM = Compose([ # Remind me to change the size for Resize
    Resize(size=(IMG_SIZE, IMG_SIZE), 
        interpolation=InterpolationMode.BILINEAR), # cv2's default
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

assert METHOD in FRS_METHODS


# Get the embeddings of the original (unperturbed) images
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


def get_embs(dataloader, original=True):
    
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
        with torch.no_grad():
            out = net(imgs.to(DEVICE))
        
        embs.append(out.detach().cpu())
    
    embs = torch.cat(embs, dim=0)
    assert (len(embs.shape) == 2) and \
        (embs.shape[0] == len(dataloader.dataset)) and \
        (embs.shape[1] == EMB_SIZE)
    
    return embs, others


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


def compute_embs(net, original=False):
    if original:
        print('Computing ORIGINAL embeddings')
        dataset = SimpleDataset(root=ORIG_IMAGES_PATH, transform=TRANSFORM)
    else:
        print('Computing NEW embeddings')
        dataset = ImageFolderWithPaths(root=DATA_PATH, transform=TRANSFORM)
        # ImageFolder will have, possibly, screwed up the mapping from class 
        # name to idx. Like: dataset.class_to_idx can be
        # {'class_552': 4552, 'class_553': 4553, 'class_554': 4554} instead of 
        # the obvious. We correct for this here:
        dataset.class_to_idx = { k : int(k.split('_')[1]) 
            for k in dataset.class_to_idx.keys() }
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=20)
    return get_embs(dataloader, original=original)


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


def get_dists(embs1, embs2, method='insightface'):
    if method == 'insightface':
        return torch.cdist(embs1, embs2)
    else:
        return 1 - torch.matmul(embs1, embs2.T)


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
dists_to_centroid = torch.gather(dists, 1, targets.view(-1,1))
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
assigns = torch.argmin(dists, axis=1)
# Prediction is: each instance is of the same class as the instance at embs[assigns]
preds = orig_labels[assigns]
correct = (preds == targets).sum().item()
n_wrong = new_embs.size(0) - correct
acc = correct / new_embs.size(0)
print(f'Accuracy is {100.*acc:3.2f}%')

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
print(troubling_ids)
import pdb; pdb.set_trace()

if PLOT:
    for tr_id in troubling_ids:
        # The anchor image for this ID
        orig_im_path = osp.join(ORIG_IMAGES_PATH, f'{tr_id}'.zfill(6) + '.jpg')
        orig_im = mpimg.imread(orig_im_path)
        # The images whose target was this ID
        paths = glob(osp.join(DATA_PATH, 'class_' + f'{tr_id}'.zfill(3), 
            '*.jpg'))
        paths = sorted(paths, key=lambda x: int(osp.basename(x).replace('.jpg', '')))
        ims = OrderedDict([ (osp.basename(p), mpimg.imread(p)) for p in paths])
        sorted_keys = sorted(ims.keys(), 
            key=lambda x: int(x.replace('.jpg', '')))
        which = torch.nonzero(targets == tr_id)
        these_preds = preds[targets == tr_id]

        fig, axes = plt.subplots(nrows=4, ncols=perts_per_id)
        # The original image
        mid_axis = axes[0, perts_per_id//2]
        mid_axis.imshow(orig_im)
        mid_axis.axis('off')
        # Each image


# Plot some troubling identities






from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5_000, random_state=0).fit(embs.detach().numpy())
print(kmeans.labels_)
print(label.numpy())
