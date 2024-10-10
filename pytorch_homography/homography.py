import torch
from pytorch_homography.ransac import build_ransac_find_homography_model


algorithms = {
    'ransac': build_ransac_find_homography_model(iterations=10000, threshold=6)
}


@torch.no_grad
def find_homography(kpt1, kpt2, algorithm='ransac'):
    if algorithm not in algorithms.keys():
        raise NotImplementedError(f'The algorithm: "{algorithm}" is not implemented yet')
    return algorithms[algorithm](kpt1, kpt2)
