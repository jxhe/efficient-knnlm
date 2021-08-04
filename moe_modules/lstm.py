import torch
import torch.nn as nn

from .mlp import LeakyReLUNet


class LSTMMOE(nn.Module):
    def __init__(self,
                 feature_size=None,
                 hidden_units=32,
                 nlayers=3,
                 dropout=0,
                 non_ctxt_dim=1024,
                 cuda=True,
                 ):
        super().__init__()

        if 'ctxt' in feature_size:
            non_ctxt_dim = feature_size['ctxt']

        non_ctxt_size = len([x for x in feature_size if x != 'ctxt'])

        if non_ctxt_size != 0:
            non_ctxt_dim = (non_ctxt_dim // non_ctxt_size) * non_ctxt_size
        else:
            non_ctxt_dim = 0

        ctxt_dim = feature_size.get('ctxt', 0)

        self.model = nn.LSTM(
            input_size=ctxt_dim + non_ctxt_dim,
            hidden_size=hidden_units,
            num_layers=nlayers,
            dropout=dropout,
            )


        self.pred = nn.Linear(hidden_units, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        input_layer = {}
        if non_ctxt_size != 0:
            ndim = non_ctxt_dim // non_ctxt_size
            for k in feature_size:
                if k != 'ctxt':
                    input_layer[k] = LeakyReLUNet(feature_size[k], ndim)

        self.input_layer = nn.ModuleDict(input_layer)

        self.feature_size = feature_size

        self.device = torch.device('cuda' if cuda else 'cpu')

        self.h = torch.zeros((nlayers, 1, hidden_units), device=self.device)
        self.c = torch.zeros((nlayers, 1, hidden_units), device=self.device)

    def forward(self, features):
        """
        Args:
            features (dict): each sub feature is of shape (seq_len, nfeat)
        """

        # import pdb; pdb.set_trace()

        features_cat = [features['ctxt'].unsqueeze(1)] if 'ctxt' in self.feature_size else []

        for k in self.feature_size:
            if k != 'ctxt':
                features_cat.append(self.input_layer[k](features[k].unsqueeze(1)))

        x = torch.cat(features_cat, -1)

        output, (h, c) = self.model(x, (self.h, self.c))

        output = self.pred(output)
        output = self.log_softmax(output)

        self.h.data, self.c.data = h.data, c.data

        return output.squeeze(1)

    def epoch_update(self):
        self.h.data.zero_()
        self.c.data.zero_()

