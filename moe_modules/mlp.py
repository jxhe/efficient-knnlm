import torch
import torch.nn as nn

class LeakyReLUNet(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.LeakyReLU(),
            nn.Linear(out_feat, out_feat),
        )

    def forward(self, features):
        return self.model(features)


class MLPMOE(nn.Module):
    def __init__(self,
                 feature_size=None,
                 hidden_units=32,
                 nlayers=3,
                 dropout=0,
                 non_ctxt_dim=512,
                 activation='relu'):
        super().__init__()

        if 'ctxt' in feature_size:
            non_ctxt_dim = feature_size['ctxt']

        non_ctxt_size = len([x for x in feature_size if x != 'ctxt'])

        if non_ctxt_size != 0:
            non_ctxt_dim = (non_ctxt_dim // non_ctxt_size) * non_ctxt_size
        else:
            non_ctxt_dim = 0

        ctxt_dim = feature_size.get('ctxt', 0)
        models = [nn.Linear(ctxt_dim + non_ctxt_dim, hidden_units), nn.Dropout(p=dropout)]
        if activation == 'relu':
            models.append(nn.ReLU())
        elif activation == 'linear':
            pass
        else:
            raise ValueError(f'activation {activation} not supported')

        for _ in range(nlayers-1):
            models.extend([nn.Linear(hidden_units, hidden_units), nn.Dropout(p=dropout)])
            if activation == 'relu':
                models.append(nn.ReLU())
            elif activation == 'linear':
                pass
            else:
                raise ValueError(f'activation {activation} not supported')

        models.append(nn.Linear(hidden_units, 2))

        models.append(nn.LogSoftmax(dim=-1))

        self.model = nn.Sequential(*models)

        input_layer = {}
        if non_ctxt_size != 0:
            ndim = non_ctxt_dim // non_ctxt_size
            for k in feature_size:
                if k != 'ctxt':
                    input_layer[k] = LeakyReLUNet(feature_size[k], ndim)

        self.input_layer = nn.ModuleDict(input_layer)

        self.feature_size = feature_size

    def forward(self, features):

        features_cat = [features['ctxt']] if 'ctxt' in self.feature_size else []

        for k in self.feature_size:
            if k != 'ctxt':
                features_cat.append(self.input_layer[k](features[k]))

        return self.model(torch.cat(features_cat, -1))


    def epoch_update(self):
        pass

