import os
import argparse
import os.path as osp
from utils.logger import setup_logger
from .gen_utils import (args2text, print_training_params, LOSS_TYPES, OPTIMS, 
    FRS_METHODS)


def parse_args():
    parser = argparse.ArgumentParser(description='Compute semantic adversaries')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum for SGD')
    parser.add_argument('--loss', type=str, default='xent', choices=LOSS_TYPES,
                        help='Loss to optimize')
    parser.add_argument('--optim', type=str, default='Adam', choices=OPTIMS,
                        help='Optimizer to use')
    parser.add_argument('--iters', type=int, default=100, 
                        help='Optimization iterations per instance')
    parser.add_argument('--lin-comb', action='store_true', default=False,
		                help='Compute adv in terms of lin. comb. of directions')
    parser.add_argument('--restarts', type=int, default=5, 
                        help='num of chunks in which to break the dataset')
    parser.add_argument('--autoattack', action='store_true', default=False,
                        help='Use FAB from AutoAttack')
    # Initialization
    parser.add_argument('--not-on-surf', action='store_true', default=False,
                        help='Random initialization is NOT on region surface')
    # Logging
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    # System to attack
    parser.add_argument('--face-recog-method', type=str, default='insightface', 
                        choices=FRS_METHODS, 
                        help='Face recognition system to use')
    # Evaluation
    parser.add_argument('--chunks', type=int, default=10, 
                        help='num of chunks in which to break the dataset')
    parser.add_argument('--num-chunk', type=int, default=None, 
                        help='index of chunk to evaluate on')
    parser.add_argument('--eval-files', action='store_true', default=False,
		                help='evaluate based on files at '
                             'exp_results/logs/results_chunk*of*_*to*.txt')
    parser.add_argument('--load-embs', action='store_true', default=False,
                        help='Whether to load embs from file')
    parser.add_argument('--embs-file', type=str, default=None,
                        help='Load embs from this file')
    # Randomization
    parser.add_argument('--seed', type=int, default=0, 
                        help='for deterministic behavior')
    args = parser.parse_args()

    args.output_dir = osp.join('exp_results', args.output_dir)

    # Log path: verify existence of output_dir dir, or create it
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    args.results_dir = osp.join(args.output_dir, 'results')
    if not osp.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    args.logs_dir = osp.join(args.output_dir, 'logs')
    if not osp.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)

    args.figs_dir = osp.join(args.output_dir, 'figs')
    if not osp.exists(args.figs_dir):
        os.makedirs(args.figs_dir, exist_ok=True)

    # txt file with all params
    chunk = 'all' if args.num_chunk is None else args.num_chunk
    args.info_log = osp.join(args.output_dir, f'info_chunk_{chunk}.txt')
    print_training_params(args, args.info_log)

    # final results
    args.final_results = osp.join(args.output_dir, f'results.txt')

    logger = setup_logger(osp.join(args.output_dir, f'chunk_{chunk}'), 
        logger_name=f'chunk_{chunk}')
    logger.info(args2text(args))
    args.LOGGER = logger

    return args