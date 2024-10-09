import torch
from torch import nn


class Ransac(nn.Module):

    def __init__(self, iterations, len_sample, model, criterion, threshold=5):
        self.iterations = iterations
        self.len_sample = len_sample
        self.model = model
        self.criterion = criterion
        self.loss_threshold = threshold

    def forward(self, x, y):
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
