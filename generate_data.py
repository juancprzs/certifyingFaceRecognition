# python3.7
"""Generates a collection of images with specified model.

Commonly, this file is used for data preparation. More specifically, before
exploring the hidden semantics from the latent space, user need to prepare a
collection of images. These images can be used for further attribute prediction.
In this way, it is able to build a relationship between input latent codes and
the corresponding attribute scores.
"""
import torch
import os.path
import argparse
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
# from models.stylegan_generator import StyleGANGenerator
from pathlib import Path
from proj_utils import set_seed
from utils.logger import setup_logger
from models.mod_stylegan_generator import ModStyleGANGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(DEVICE, seed=2)

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Generate images with given model.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images to generate. This field will be '
                           'ignored if `latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('-S', '--generate_style', action='store_true',
                      help='If specified, will generate layer-wise style codes '
                           'in Style GAN. (default: do not generate styles)')
  parser.add_argument('-I', '--generate_image', action='store_false',
                      help='If specified, will skip generating images in '
                           'Style GAN. (default: generate images)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(args.output_dir, logger_name='generate_data')
  # Images dir
  ims_out_dir = os.path.join(args.output_dir, 'ims')
  Path(ims_out_dir).mkdir(parents=True, exist_ok=True)
  # Tensors dir
  tens_out_dir = os.path.join(args.output_dir, 'tensors')
  Path(tens_out_dir).mkdir(parents=True, exist_ok=True)

  logger.info(f'Initializing generator.')
  gan_type = MODEL_POOL[args.model_name]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(args.model_name, logger)
    kwargs = {}
  elif gan_type == 'stylegan':
    model = ModStyleGANGenerator(args.model_name, logger)
    kwargs = {'latent_space_type': args.latent_space_type}
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.latent_codes_path):
    logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
    latent_codes = np.load(args.latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
  else:
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(args.num, **kwargs)
  total_num = latent_codes.shape[0]

  logger.info(f'Generating {total_num} samples.')
  results = defaultdict(list)
  pbar = tqdm(total=total_num, leave=False)
  for latent_codes_batch in model.get_batch_inputs(latent_codes):
    if gan_type == 'pggan':
      outputs = model.easy_synthesize(latent_codes_batch)
    elif gan_type == 'stylegan':
      outputs = model.easy_synthesize(latent_codes_batch,
                                      **kwargs,
                                      generate_style=args.generate_style,
                                      generate_image=args.generate_image)
    for key, val in outputs.items():
      if key == 'image':
        for image in val:
          # Save tensor
          save_path_tens = os.path.join(tens_out_dir, f'{pbar.n:06d}.pth')
          # torch.save(image.detach().cpu(), save_path_tens)
          # And the image as PNG
          save_path_png = os.path.join(ims_out_dir, f'{pbar.n:06d}.png')
          curr_image = 255. * image.cpu().detach().numpy().transpose(1, 2, 0)
          cv2.imwrite(save_path_png, curr_image[:, :, ::-1])
          pbar.update(1)
      else:
        results[key].append(val)
    if 'image' not in outputs:
      pbar.update(latent_codes_batch.shape[0])
    if pbar.n % 1000 == 0 or pbar.n == total_num:
      logger.debug(f'  Finish {pbar.n:6d} samples.')
  pbar.close()

  logger.info(f'Saving results.')
  for key, val in results.items():
    save_path = os.path.join(args.output_dir, f'{key}.npy')
    np.save(save_path, np.concatenate(val, axis=0))


if __name__ == '__main__':
  main()
