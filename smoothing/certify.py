import argparse
from time import time
import torch
import datetime
from tqdm import tqdm

from .certificate import L2Certificate
from .smooth import Smooth


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
        "--anisortopic-sigma-path", type=str, default=None,
        help="Path to Anisotropic Sigma that can be used in certification"
    )

    # dataset options
    parser.add_argument(
        "--latent-path", type=str, default=None, required=True,
        help="dataset folder path, required for ImageNet"
    )
    parser.add_argument(
        "--directions-path", type=str, default=None, required=True,
        help="dataset folder path, required for ImageNet"
    )
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

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from .model import wraped_model

    # load the base classifier
    directions = torch.load(args.directions_path)
    model = wraped_model(directions, args.face_recog_model)

    # get the dataset
    dataset = torch.load(args.latent_path)    

    # get the type of certificate
    certificate = L2Certificate(1, device=device)

    # Loading sigma for smoothing
    if args.anisortopic_sigma_path is None:
        sigma = torch.tensor([args.sigma], device=device)
    else:
        sigma = torch.load(args.anisortopic_sigma_path).to(device)

    # prepare output file
    f = open(args.outfile, 'w')
    print(
        "idx\tlabel\tpredict\tcorrect\tgap\tradius\ttime",
        file=f,
        flush=True
    )

    # Number of identities is the number of latents
    num_classes = dataset.shape[0]
    # Getting the number of directions we are going to certify
    number_of_dirs = directions.shape[0]
    # Initializing the perturbation along these directions to 0
    x = torch.zeros((1, number_of_dirs), device=device)
    
    for i in tqdm(range(num_classes)):
        # only certify every args.skip examples, and stop after args.max
        # examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        z, label = dataset[i].to(device), torch.tensor([i], device=device)

        before_time = time()

        # certify the point
        smoothed_classifier = Smooth(model, num_classes, sigma, certificate)
        prediction, gap = smoothed_classifier.certify(
            z, x, args.N0, args.N, args.alpha, args.batch_sz
        )
        after_time = time()

        # compute radius
        correct = int(prediction == label)

        # I am adding .min here in case of anisotropic guy
        radius = sigma.min().item() * gap

        time_elapsed = str(datetime.timedelta(
            seconds=(after_time - before_time)))
        print(
            "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{}".format(
                i,
                label,
                prediction,
                correct,
                gap,
                radius,
                time_elapsed),
            file=f,
            flush=True
        )
    f.close()