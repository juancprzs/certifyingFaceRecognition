import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.mod_stylegan_generator import ModStyleGANGenerator
from attack_utils.gen_utils import (get_transform, lat2embs,
     get_latent_codes,  INP_RESOLS, MEAN, STD, DEVICE, EMB_SIZE)
from main_attack import get_net, get_transform


class wraped_model(nn.Module):
    def __init__(self, direction_matrix, face_recog='insightface') -> None:
        super().__init__()
        
        # Loading the GAN
        Generator = ModStyleGANGenerator('stylegan_ffhq')
        Generator.model.eval().to(DEVICE)
        self.generator = Generator.easy_synthesize

        # Loading the face recognition guy
        self.face_reco = get_net(face_recog).to(DEVICE)

        # Needed transformations for the generated images.
        self.transform = get_transform(INP_RESOLS[face_recog], MEAN, STD)

        # The matrix for semantic directions
        self.proj_matrix = direction_matrix

        # Getting the latents for our identities.
        latents = get_latent_codes(Generator).to(DEVICE)

        # Computing original embeddings for these identities.
        print("Computing original embeddings.")
        self.original_embeddings = lat2embs(Generator, self.face_reco, latents,
                self.transform, few=False, with_tqdm=True, return_ims=False)

    def compute_probs(self, embedding):
        #Compute the probability based on distances.
        all_dists = torch.cdist(embedding, self.original_embeddings, 
                                compute_mode='donot_use_mm_for_euclid_dist')
        all_dists = all_dists / np.sqrt(EMB_SIZE)
        return F.softmax(-all_dists, dim=1)

    def forward(self, x, p=0):
        # Adding perturbations to the latent
        x = x + torch.matmul(self.proj_mat, p)
        # Generate the face
        x = self.generator(x)
        # Transform (Resize)
        x = self.transform(x)
        # Compute Embedding
        x = self.face_reco(x)
        # Get probability vector
        return  self.compute_probs(x)