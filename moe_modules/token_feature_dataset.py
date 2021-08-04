import torch


class TokenFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, ctxt_dataset, other_dataset, kenlm_dataset=None, ngram=0):
        super().__init__()

        self.ctxt_dataset = ctxt_dataset
        self.other_dataset = other_dataset
        self.kenlm_dataset = kenlm_dataset

        if ngram == 0 and self.other_dataset:
            self.ngram = len(self.other_dataset[0]['freq'])
        else:
            self.ngram = ngram

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index):

        local_f = []

        hypo_ctxt = self.ctxt_dataset[index] if self.ctxt_dataset else None
        hypo_other = self.other_dataset[index] if self.other_dataset else None

        return{
            "id": index,
            "ctxt": hypo_ctxt['ctxt'] if hypo_ctxt else None,
            "lm_ent": [hypo_other['lm_ent']] if hypo_other else None,
            "lm_max": [hypo_other['lm_max']] if hypo_other else None,
            "freq": hypo_other['freq'] if hypo_other else None,
            "fert": hypo_other['fert'] if hypo_other else None,
            "lm_scores": hypo_other['lm_s'] if hypo_other else None,
            "knn_scores": self.kenlm_dataset[index]['kenlm_s'] if self.kenlm_dataset is not None else hypo_other['knn_s'],
        }

    def __len__(self):
        try:
            return len(self.ctxt_dataset)
        except:
            return len(self.other_dataset)

    def collater(self, samples):
        def merge(key, dtype=torch.float32):
            if len(samples) == 0 or samples[0][key] is None:
                return None

            return torch.tensor([s[key] for s in samples],
                                dtype=dtype,
                                device=self.device)

        batch = {
            'feature':{
                'ctxt': merge('ctxt'),
                'lm_ent': merge('lm_ent'),
                'lm_max': merge('lm_max'),
                'freq': merge('freq'),
                'fert': merge('fert'),
            },
            'id': merge('id'),
            'lm_scores': merge('lm_scores'),
            'knn_scores': merge('knn_scores'),
        }
        return batch

    def get_nfeature(self, feature_name):
        return len(self.__getitem__(0)[feature_name])
