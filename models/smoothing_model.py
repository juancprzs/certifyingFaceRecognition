import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.mod_stylegan_generator import ModStyleGANGenerator
from attack_utils.gen_utils import (get_transform, lat2embs, get_latent_codes, 
    INP_RESOLS, MEAN, STD, EMB_SIZE)
from main_attack import get_net, get_transform


class WrappedModel(nn.Module):
    def __init__(self, direction_matrix, face_recog='insightface', 
            embs_file=None) -> None:
        super().__init__()
        
        # Loading the GAN
        self.device = direction_matrix.device
        self.generator = ModStyleGANGenerator('stylegan_ffhq')
        self.generator.model.eval().to(self.device)

        # Loading the face recognition guy
        self.face_recog = face_recog
        self.face_reco = get_net(self.face_recog).to(self.device)

        # Needed transformations for the generated images.
        self.transform = get_transform(INP_RESOLS[self.face_recog], MEAN, STD)

        # Matrix of semantic directions: Dim=[num_directions, latent_dimension]
        self.dir_mat = direction_matrix

        # Getting the latents for our identities.
        self.latents = get_latent_codes(self.generator).to(self.device)

        # Computing original embeddings for these identities.
        if embs_file is None:
            print('Computing original embeddings')
            self.orig_embs = lat2embs(self.generator, self.face_reco, 
                self.latents, self.transform, few=False, with_tqdm=True, 
                return_ims=False)[0]
            print(f'Computed {self.orig_embs.size(0)} embeddings')
        else:
            print(f'Loading original embeddings from "{embs_file}"')
            self.orig_embs = torch.load(embs_file)
            print(f'Loaded {self.orig_embs.size(0)} embeddings')


    def compute_probs(self, embedding):
        # Compute the probability based on distances.
        all_dists = torch.cdist(embedding, self.orig_embs, 
                                compute_mode='donot_use_mm_for_euclid_dist')
        all_dists = all_dists / np.sqrt(EMB_SIZE)
        return F.softmax(-all_dists, dim=1)

    def forward(self, x, p=0):
        # Adding perturbations to the latent
        p = p.squeeze(2).squeeze(1)
        pert = p @ self.dir_mat # torch.matmul(p, self.dir_mat)
        x = x + pert
        embs, _ = lat2embs(self.generator, self.face_reco, x, self.transform, 
            few=False)
        # Get probability vector
        probs = self.compute_probs(embs.cpu()).to(self.device)
        return probs
