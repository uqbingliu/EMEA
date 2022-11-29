# -*- coding: utf-8 -*-

import numpy as np
import torch
from emea.conf import Config
from tqdm import tqdm, trange


class ConflictIndicator:
    def __init__(self, conf: Config):
        self.conf = conf
        self.device = torch.device("cuda:0")

    def construct_factor_graphs(self, embs: np.ndarray):
        ent_num, _ = embs.shape
        ent_to_factors_map = dict()
        with torch.no_grad():
            embs = torch.tensor(embs, device=self.device)
            simi_mtx = torch.matmul(embs, embs.t())
            simi_mtx.fill_diagonal_(-1.0)
            batch_size = 1024
            sub_rank_mtx_list = []
            for cur in range(0, ent_num, batch_size):
                sub_simi_mtx = simi_mtx[cur: cur+batch_size]
                sub_rank_mtx = torch.argsort(sub_simi_mtx, dim=1, descending=True)
                sub_rank_mtx_list.append(sub_rank_mtx.cpu().numpy())
            # rank_mtx = torch.argsort(simi_mtx, dim=1, descending=True)
            rank_mtx = np.concatenate(sub_rank_mtx_list, axis=0)
            nb_mtx = rank_mtx[:, 0:self.conf.conflict_checker_neigh_num]
        nb_mtx = nb_mtx.tolist()
        for idx in trange(len(nb_mtx)):
            ents = [idx] + nb_mtx[idx]
            for ent in ents:
                if ent not in ent_to_factors_map:
                    ent_to_factors_map[ent] = [idx]
                else:
                    ent_to_factors_map[ent].append(idx)
        self.ent_to_factors_map = {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in ent_to_factors_map.items()}
        self.nb_mtx = torch.tensor(nb_mtx, device=self.device)

    def apply_checker(self, alignment: np.ndarray, prob_mtx: np.ndarray):
        with torch.no_grad():
            alignment = torch.tensor(alignment, device=self.device)
            prob_mtx = torch.tensor(prob_mtx, device=self.device)
            ent1_arr = alignment[:, 0]
            ent2_arr = alignment[:, 1]
            labels = - torch.ones(size=(prob_mtx.shape[0],), dtype=torch.long, device=self.device)
            probs = torch.zeros(size=(prob_mtx.shape[0],), dtype=torch.float32, device=self.device)
            labels[ent1_arr] = ent2_arr
            tmp = torch.gather(prob_mtx[ent1_arr], dim=1, index=ent2_arr.unsqueeze(dim=1))
            probs[ent1_arr] = tmp.reshape(shape=ent1_arr.shape)
            ent1_nb_mtx = self.nb_mtx[ent1_arr]
            ent1_nb_labels = labels[ent1_nb_mtx]
            ent1_nb_probs = probs[ent1_nb_mtx]
            ent1_probs = probs[ent1_arr]

            eq_label_mtx = torch.eq(ent2_arr.unsqueeze(dim=1), ent1_nb_labels).to(dtype=torch.long)
            lt_prob_mtx = (ent1_probs.unsqueeze(dim=1) < ent1_nb_probs).to(dtype=torch.long)
            both_cond_mtx = eq_label_mtx * lt_prob_mtx
            overall_arr = (torch.sum(both_cond_mtx, dim=1, keepdim=False) >= 1).to(dtype=torch.long)
            conf_arr = (torch.sum(eq_label_mtx, dim=1, keepdim=False) >= 1).to(dtype=torch.long)
        return conf_arr, overall_arr

    def apply_checker2(self, ent1_arr: np.ndarray, candi_mtx: np.ndarray, sampled_alignment: np.ndarray, prob_mtx: np.ndarray):
        with torch.no_grad():
            ent1_arr = torch.tensor(ent1_arr, device=self.device)
            candi_mtx = torch.tensor(candi_mtx, device=self.device)
            sampled_alignment = torch.tensor(sampled_alignment, device=self.device)
            prob_mtx = torch.tensor(prob_mtx, device=self.device)

            sampled_ent1_arr = sampled_alignment[:, 0]
            sampled_ent2_arr = sampled_alignment[:, 1]
            labels = - torch.ones(size=(prob_mtx.shape[0],), dtype=torch.long, device=self.device)
            probs = torch.zeros(size=(prob_mtx.shape[0],), dtype=torch.float32, device=self.device)
            labels[sampled_ent1_arr] = sampled_ent2_arr
            tmp = torch.gather(prob_mtx[sampled_ent1_arr], dim=1, index=sampled_ent2_arr.unsqueeze(dim=1))
            probs[sampled_ent1_arr] = tmp.reshape(shape=sampled_ent1_arr.shape)
            ent1_nb_mtx = self.nb_mtx[ent1_arr]
            ent1_nb_labels = labels[ent1_nb_mtx]
            ent1_nb_probs = probs[ent1_nb_mtx]

            candi_probs = torch.gather(prob_mtx[ent1_arr], index=candi_mtx, dim=1)
            candi_mtx = candi_mtx.transpose(0, 1).unsqueeze(dim=2)
            candi_probs = candi_probs.transpose(0, 1).unsqueeze(dim=2)

            eq_label_mtx = torch.eq(candi_mtx, ent1_nb_labels.unsqueeze(dim=0)).to(dtype=torch.long)
            overall_mtx = (torch.sum(eq_label_mtx, dim=-1, keepdim=False) >= 1).to(dtype=torch.long)
            overall_mtx = overall_mtx.transpose(0, 1)
        return overall_mtx

    def set_embs_n_probs(self, embs, prob_mtx):
        self.prob_mtx = torch.tensor(prob_mtx, device=self.device)
        self.construct_factor_graphs(embs)

    def factor_func(self, fidx_arr: torch.Tensor, all_labels: torch.Tensor, target_ent=None):
        with torch.no_grad():
            fidx_ent1_nb_mtx = self.nb_mtx[fidx_arr]
            fidx_ent1_nb_labels = all_labels[fidx_ent1_nb_mtx]
            fidx_ent1_arr = fidx_arr
            fidx_ent1_labels = all_labels[fidx_ent1_arr]
            eq_label_mtx = torch.eq(fidx_ent1_labels.unsqueeze(dim=1), fidx_ent1_nb_labels).to(dtype=torch.long)
            conflict_arr = (torch.sum(eq_label_mtx, dim=1, keepdim=False) >= 1).to(dtype=torch.long)  # have conficts

            # opt1: get 1 if no conflict,
            scores = 1.0 - conflict_arr  # opt1: get 1 if no conflict, scores on few factor graphs

            # opt 2
            # prob = self.prob_mtx[fidx_arr, fidx_ent1_labels]
            # scores = torch.pow(self.conf.cf_alpha1, (conflict_arr >= 1).to(dtype=torch.float32)) * prob

            # opt 3
            # fidx_ent1_probs = self.prob_mtx[fidx_arr, fidx_ent1_labels]
            # flat_fidx_ent1_nb = fidx_ent1_nb_mtx.reshape(-1)
            # # flat_fidx_ent1_nb_labels = fidx_ent1_labels.unsqueeze(dim=1).repeat(1, fidx_ent1_nb_mtx.shape[1]).reshape(-1)
            # flat_fidx_ent1_nb_labels = fidx_ent1_nb_labels.reshape(-1)
            # fidx_ent1_nb_probs = self.prob_mtx[flat_fidx_ent1_nb, flat_fidx_ent1_nb_labels].reshape(fidx_ent1_nb_mtx.shape)
            # lt_prob_mtx = (fidx_ent1_probs.unsqueeze(dim=1) < fidx_ent1_nb_probs).to(dtype=torch.long)
            # eq_n_lt_prob_mtx = eq_label_mtx * lt_prob_mtx
            # conf_lt_prob_arr = (torch.sum(eq_n_lt_prob_mtx, dim=1, keepdim=False) >=1).to(dtype=torch.long)
            # conf_lt_score = 1.0 - conf_lt_prob_arr
            # scores = torch.ones(size=fidx_arr.shape, device=self.device)
            # scores[conflict_arr >= 1] = self.conf.cf_alpha2
            # scores[eq_n_lt_prob_arr >= 1] = self.conf.cf_alpha1
            # scores = scores * fidx_ent1_probs
            # scores = torch.pow(0.8, (eq_n_lt_prob_arr >= 1).to(dtype=torch.float32)) * fidx_ent1_probs

        return scores

    def compute_factors_involving_entity(self, ent, all_labels, target_ent=None):
        related_factors = self.ent_to_factors_map[ent]
        score_arr = self.factor_func(related_factors, all_labels, target_ent)
        sum_score = torch.sum(score_arr)
        return sum_score

    def compute_factors_in_batch(self, ent_arr: np.ndarray, candi_mtx:np.ndarray, sampled_alignment:list, neu_pred_alignment: list):
        sampled_alignment = torch.tensor(sampled_alignment, device=self.device)
        pred_alignment_map = dict(neu_pred_alignment)
        all_labels = - torch.ones(size=(self.prob_mtx.shape[0],), device=self.device, dtype=torch.long)
        all_labels[sampled_alignment[:, 0]] = sampled_alignment[:, 1]
        fea_shape = list(candi_mtx.shape)
        fea_shape.append(2)
        fea_mtx = torch.zeros(size=tuple(fea_shape), dtype=torch.float32, device=self.device)

        for idx1, ent1 in tqdm(list(enumerate(ent_arr))):
            for idx2, ent2 in enumerate(candi_mtx[idx1]):
                ori_ent2 = all_labels[ent1]
                all_labels[ent1] = ent2
                fea = self.compute_factors_involving_entity(ent1, all_labels, ent2)
                all_labels[ent1] = ori_ent2
                fea_mtx[idx1][idx2][0] = fea
                fea2 = torch.eq(ent2, pred_alignment_map[ent1]).to(dtype=torch.float32)
                fea_mtx[idx1][idx2][1] = fea2
        return fea_mtx


    def compute_factors_in_batch2(self, ent_arr: np.ndarray, candi_mtx:np.ndarray, sampled_alignment:list):
        sampled_alignment = torch.tensor(sampled_alignment, device=self.device)
        all_labels = - torch.ones(size=(self.prob_mtx.shape[0],), device=self.device, dtype=torch.long)
        all_labels[sampled_alignment[:, 0]] = sampled_alignment[:, 1]
        fea_mtx = torch.zeros(size=candi_mtx.shape, dtype=torch.float32, device=self.device)

        for idx1, ent1 in tqdm(list(enumerate(ent_arr))):
            for idx2, ent2 in enumerate(candi_mtx[idx1]):
                ori_ent2 = all_labels[ent1]
                all_labels[ent1] = ent2
                fea = self.compute_factors_involving_entity(ent1, all_labels)
                all_labels[ent1] = ori_ent2
                # fea = torch.pow(0.8, (fea >= 1).to(dtype=torch.float32)) * self.prob_mtx[ent1][ent2]
                fea = torch.pow(0.8, fea) * self.prob_mtx[ent1][ent2]
                fea_mtx[idx1][idx2] = fea
        return fea_mtx


