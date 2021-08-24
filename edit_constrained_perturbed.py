# python3.7
"""Edits latent codes with respect to given boundary.

Basically, this file takes latent codes and a semantic boundary as inputs, and
then shows how the image synthesis will change if the latent codes is moved
towards the given boundary.

NOTE: If you want to use W or W+ space of StyleGAN, please do not randomly
sample the latent code, since neither W nor W+ space is subject to Gaussian
distribution. Instead, please use `generate_data.py` to get the latent vectors
from W or W+ space first, and then use `--input_latent_codes_path` option to
pass in the latent vectors.
"""

import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate

# Local imports
import os.path as osp
from proj_utils import (get_projection_matrices, sample_ellipsoid, sq_distance,
  project_to_region, DATASETS, GAN_NAMES, ATTRS)

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary by '\
          'sampling within ellipse.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-d', '--dataset', type=str, default='ffhq',
                      help='Name of dataset (default: ffhq)', choices=DATASETS)
  parser.add_argument('-g', '--gan_name', type=str, default='stylegan',
                      help='Name of GAN (default: stylegan)', choices=GAN_NAMES)
  parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images for editing. This field will be '
                           'ignored if `input_latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('--samples', type=int, default=10,
                      help='Number of steps for image editing. (default: 10)')
  parser.add_argument('--sample_surface', action='store_true',
                      help='Sample from surface of hyper-ellipsoid of interest')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  # Get our projection matrices
  proj_mat, ellipse_mat, dirs, files = get_projection_matrices(
    dataset=args.dataset, gan_name=args.gan_name
  )

  logger = setup_logger(args.output_dir, logger_name='generate_data')

  logger.info(f'Initializing generator.')
  gan_type = MODEL_POOL[args.model_name]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(args.model_name, logger)
    kwargs = {}
  elif gan_type == 'stylegan':
    model = StyleGANGenerator(args.model_name, logger)
    kwargs = {'latent_space_type': args.latent_space_type}
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

  logger.info(f'Preparing boundary.')
  for boundary_file in files:
    boundary = np.load(boundary_file)
    basename = osp.basename(boundary_file).replace('.npy', '')
    new_filename = osp.join(args.output_dir, f'boundary_{basename}.npy')
    np.save(new_filename, boundary)

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.input_latent_codes_path):
    logger.info(f'  Load latent codes from `{args.input_latent_codes_path}`.')
    latent_codes = np.load(args.input_latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
  else:
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(args.num, **kwargs)
  np.save(os.path.join(args.output_dir, 'latent_codes.npy'), latent_codes)
  total_num = latent_codes.shape[0]

  logger.info(f'Editing {total_num} samples.')
  for sample_id in tqdm(range(total_num), leave=False):
    # # # #                       From here
    # Get the deltas
    ellipse_points = sample_ellipsoid(ellipse_mat, n_vecs=args.samples)
    # Project points inside ellipse to subspace spanned by our directions
    inside_ellipse, inside_dirs = project_to_region(ellipse_points, proj_mat, 
        ellipse_mat, check=True, dirs=dirs, on_surface=args.sample_surface)
    if not args.sample_surface:
      assert np.allclose(inside_ellipse, inside_dirs), 'Should be equal, as '\
        'they were sampled within ellipse'
    deltas = inside_ellipse.T # 'deltas' is of shape [args.samples, n_dims]
    class_path = osp.join(args.output_dir, 'data', f'class_{sample_id:03d}')
    os.makedirs(class_path)
    # # # #                       To here
    new_codes = latent_codes[sample_id:sample_id + 1] + deltas
    interpolation_id = 0
    for interpolations_batch in model.get_batch_inputs(new_codes):
      if gan_type == 'pggan':
        outputs = model.easy_synthesize(interpolations_batch)
      elif gan_type == 'stylegan':
        outputs = model.easy_synthesize(interpolations_batch, **kwargs)
      for image in outputs['image']:
        save_path = osp.join(class_path, f'{interpolation_id:03d}.jpg')
        # from glob import glob
        # import pdb; pdb.set_trace()
        cv2.imwrite(save_path, image[:, :, ::-1])
        interpolation_id += 1
    assert interpolation_id == args.samples
    logger.debug(f'  Finished sample {sample_id:3d}.')
  logger.info(f'Successfully edited {total_num} samples.')


if __name__ == '__main__':
  main()
