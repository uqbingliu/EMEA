# -*- coding: utf-8 -*-

import numpy as np
import torch
from emea.conf import Config
from tqdm import tqdm, trange


class FollowNeuIndicator:
    def __init__(self, conf: Config):
        self.conf = conf
        self.device = torch.device("cuda:0")

    def compute_factors_in_batch(self, ent_arr: np.ndarray, candi_mtx:np.ndarray, neu_pred_alignment: list):
        pred_alignment_map = dict(neu_pred_alignment)
        fea_mtx = torch.zeros(size=candi_mtx.shape, dtype=torch.float32, device=self.device)

        for idx1, ent1 in tqdm(list(enumerate(ent_arr))):
            for idx2, ent2 in enumerate(candi_mtx[idx1]):
                fea2 = torch.eq(ent2, pred_alignment_map[ent1]).to(dtype=torch.float32)
                fea_mtx[idx1][idx2] = fea2
        return fea_mtx



