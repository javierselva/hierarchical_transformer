# Code copied and adapted from https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

import torch
from architecture.ht import *

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICReg(nn.Module):
    def __init__(self, config):
        super(VICReg,self).__init__()
        self.backbone = HierarchicalTransformer(config)
        # Grab info from the last group of the backbone
        architecture = config['arch']['architecture_' + self.backbone.groups[-1]]
        pred_h_dim = architecture['pred_h_dim']
        if pred_h_dim is not None:
            if isinstance(pred_h_dim, int):
                pred_h_dim = [pred_h_dim]
            elif isinstance(pred_h_dim, str):
                pred_h_dim = eval(pred_h_dim)
        else:
            pred_h_dim = []
        # TODO allow to specify different out dim for comparisons
        self.projector = Predictor(d_in=architecture['dim'],
                                   d_out=architecture['dim'],
                                   D=architecture['nD'],
                                   d_mid=pred_h_dim)

        self.sim_coeff = config['train']['loss_params']['VICREG']['sim_coeff']
        self.std_coeff = config['train']['loss_params']['VICREG']['std_coeff']
        self.cov_coeff = config['train']['loss_params']['VICREG']['cov_coeff']
        self.batch_size = config['train']['batch_size']
        self.num_features = self.backbone.final_output_dim

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        # This is used to gather the features from all the GPUs (not our case)
        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

    def train(self, mode=Stage.TRAIN):
        self.training = mode
        self.backbone.train(mode)
        self.projector.train(mode == Stage.TRAIN)

    def eval(self):
        self.training = False
        self.backbone.eval()
        self.projector.eval()

    def get_groups(self):
        return self.backbone.groups

    def get_params(self, named=False):
        if named:
            return list(self.backbone.named_parameters()), list(self.projector.named_parameters())
        else:
            return list(self.backbone.parameters()), list(self.projector.parameters())