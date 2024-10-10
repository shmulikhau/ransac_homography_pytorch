import torch
from pytorch_homography.ransac import build_ransac_find_homography_model


algorithms = {
    'ransac': build_ransac_find_homography_model(iterations=15000, threshold=6)
}


@torch.no_grad
def find_homography(kpt1, kpt2, algorithm='ransac'):
    """
    find homography from tow sets of key-points.

    Parameters:
    kpt1 (torch.tensor): set a of key-points as shape of b,2.
    kpt2 (torch.tensor): set b of key-points as shape of b,2.
    algorithm (str): which algorithm to use, default ransac.

    Returns:
    a tensor of homography matrix in size of 3x3.
    """
    if algorithm not in algorithms.keys():
        raise NotImplementedError(f'The algorithm: "{algorithm}" is not implemented yet')
    return algorithms[algorithm](kpt1, kpt2)
