import os
import torch
import argparse
import datetime
import os.path as osp
from time import time
from tqdm import tqdm

from smoothing.smooth import Smooth
from smoothing.certificate import L2Certificate
from models.smoothing_model import WrappedModel
from attack_utils.gen_utils import get_latent_codes, get_all_matrices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Certify face recognition examples')

    parser.add_argument(
        "--face-recog-model", required=True,
        choices=["insightface", "facenet"],
        type=str, help="type of model to load for face recognition"
    )
    parser.add_argument(
        "--outfile", required=True,
        type=str, help="output csv file"
    )
    parser.add_argument(
        "--sigma", type=float, required=True,
        help="noise hyperparameter, required for initialization " +
        "in isotropic_dd and ancer"
    )
    parser.add_argument(
        "--anisotropic-sigma", action='store_true', default=False,
        help="Whether to use Anisotropic Sigma for certification"
    )

    # dataset options
    parser.add_argument(
        "--skip", type=int, default=1,
        help="skip examples in the dataset"
    )
    parser.add_argument(
        "--max", type=int, default=-1,
        help="stop after a certain number of examples"
    )

    # certification parameters
    parser.add_argument(
        "--batch-sz", type=int,
        default=100, help="certification batch size"
    )
    parser.add_argument(
        "--N0", type=int, default=100
    )
    parser.add_argument(
        "--N", type=int, default=100000,
        help="number of samples to use"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.001,
        help="failure probability"
    )
    parser.add_argument('--load-n-embs', type=int, default=1_000_000, 
        help='num of embs. Default is all of them (1M)')

    args = parser.parse_args()


    # Load the matrix of directions
    dirs = get_all_matrices()[3].T

    device = dirs.device
    # Instantiate the wrapped model
    model = WrappedModel(dirs, args.face_recog_model, n_embs=args.load_n_embs,
        load_embs=True, embs_file=None)
    # Get the dataset of latent codes
    dataset = model.latents
    dataset = dataset.to(device)

    # get the type of certificate
    certificate = L2Certificate(1, device=device)

    # Loading sigma for smoothing
    if args.anisotropic_sigma:
        # Read the inverse of the matrix that parameterizes the ellipse
        red_ellipse_mat_inv = get_all_matrices()[6]
        # This (the *inverse) is equivalent to the sigma for smoothing
        sigma = torch.diag(red_ellipse_mat_inv).to(device)
        # sigma = torch.load(args.anisotropic_sigma_path).to(device)
        # Scale matrix by (scalar) sigma argument
        sigma = args.sigma * sigma
    else:
        sigma = torch.tensor([args.sigma], device=device)

    # prepare output file
    parent_dir = osp.dirname(args.outfile)
    if not osp.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    with open(args.outfile, 'w+') as f:
        print(
            "idx\tlabel\tpredict\tcorrect\tgap\tradius\ttime",
            file=f,
            flush=True
        )

    # Number of identities is the number of latents
    num_classes = dataset.shape[0]
    print(f'Found {num_classes} classes')
    # Getting the number of directions we are going to certify
    num_dirs = dirs.shape[0]
    print(f'Found {num_dirs} directions')
    # Initializing the perturbation along these directions to 0
    x = torch.zeros((1, num_dirs), device=device)
    
    smoothed_classifier = Smooth(model, num_classes, sigma, certificate)
    
    for i in tqdm(range(num_classes)):
        # Only certify every args.skip examples. Stop after args.max examples
        if (i + 1) % args.skip != 0:
            continue
        if (i + 1) == args.max:
            break

        z, label = dataset[i].to(device), torch.tensor([i], device=device)

        before_time = time()

        # certify the point 
        prediction, gap = smoothed_classifier.certify(
            z.unsqueeze(0), x, label, args.N0, args.N, args.alpha, args.batch_sz
        )
        after_time = time()

        # compute radius
        correct = int(prediction == label)

        # I am adding .min here for the anisotropic case
        radius = sigma.min().item() * gap

        time_elapsed = str(datetime.timedelta(
            seconds=(after_time - before_time)))
        with open(args.outfile, 'a') as f:
            print(
                "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{}".format(
                    i,
                    label.item(),
                    prediction,
                    correct,
                    gap,
                    radius,
                    time_elapsed),
                file=f,
                flush=True
            )
