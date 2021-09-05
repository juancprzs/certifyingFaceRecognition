import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize

class wraped_model(nn.Module):
    def __init__(self, generator, face_reco, projection_matrix) -> None:
        super().__init__()

        self.generator = generator.easy_synthesize
        self.face_reco = face_reco
        self.transform = get_transform()
        self.proj_mat = projection_matrix

    def compute_probs(self, embedding):
        #Compute the probability based on distances.
        return

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




def get_transform():
    def resize(x):
        x = x.unsqueeze(0) if x.ndim != 4 else x
        interp = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', 
            align_corners=False)
        return interp.squeeze(0)

    normalize = Normalize((MEAN, MEAN, MEAN), (STD, STD, STD))
    return Compose([resize, normalize])
# Transforms
