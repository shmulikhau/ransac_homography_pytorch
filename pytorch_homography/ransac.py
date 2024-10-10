import torch
from torch import nn
from pytorch_homography.homography_utils import get_homography, distance_vectors


class Ransac(nn.Module):
    """
    Ransac (Random sample consensus) algorithm implemention on torch
    """

    def __init__(self, iterations, len_sample, model, criterion, threshold=5):
        super(Ransac, self).__init__()
        self.iterations = iterations
        self.len_sample = len_sample
        self.model = model
        self.criterion = criterion
        self.loss_threshold = threshold

    def forward(self, x, y):
        """
        forward function

        Parameters:
        x (torch.tensor): tensor of a set as size of b,n
        y (torch.tensor): tensor of b set as size of b,n

        Rertuns:
        ransac result, how mutch data was corresponding
        """
        device = x.device
        selections = torch.rand(self.iterations, self.len_sample, device=device)
        selections = selections * (len(x) - 1e-8)
        selections = selections.type(torch.int32)
        # sel-dim=epochs,4,2
        x_sel = torch.index_select(x,0,selections.reshape(-1)).reshape(self.iterations, self.len_sample, 2)
        y_sel = torch.index_select(y,0,selections.reshape(-1)).reshape(self.iterations, self.len_sample, 2)
        all_models = self.model(x_sel, y_sel)
        lose_arr = self.criterion(all_models, x, y)
        lose_arr = (lose_arr < self.loss_threshold).sum(dim=-1)
        #lose_arr = -lose_arr.sum(dim=-1)
        return all_models[torch.argmax(lose_arr)], torch.max(lose_arr)


def build_ransac_find_homography_model(iterations=10000, threshold=6):
    return Ransac(iterations, 4, get_homography, distance_vectors, threshold=threshold)
