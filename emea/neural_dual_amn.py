# -*- coding: utf-8 -*-

from emea.emea_framework import NeuralEAModule
from emea.conf import Config
import os
import numpy as np
import json
import torch
from emea.simi_to_prob import SimiToProbModule
from emea.data import load_alignment
from emea.Dual_AMN.runner import Runner
import subprocess
from emea.RREA.CSLS_torch import Evaluator


class DualAMNModule(NeuralEAModule):
    def __init__(self, conf: Config):
        super(DualAMNModule, self).__init__(conf)

    # def prepare_data(self):
    #     pass

    def train_model_with_observed_labels(self):

        if self.conf.py_exe_fn is None:
            runner = Runner(self.conf.data_dir, self.conf.output_dir,
                            max_train_epoch=self.conf.max_train_epoch,
                            max_continue_epoch=self.conf.max_continue_epoch)
            if self.conf.initial_training == "supervised" or self.conf.initial_training == "sup":
                runner.train()
            elif self.conf.initial_training == "iterative" or self.conf.initial_training == "semi" :
                runner.iterative_training()
            else:
                raise Exception("unknown initial training method")
            runner.save(save_metrics=self.conf.neu_save_metrics)
        else:
            cmd_fn = self.conf.py_exe_fn
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            script_fn = os.path.join(cur_dir, "Dual_AMN/runner.py")
            args_str = f"--data_dir={self.conf.data_dir} --output_dir={self.conf.output_dir} " \
                       f"--max_train_epoch={self.conf.max_train_epoch} --max_continue_epoch={self.conf.max_continue_epoch} " \
                       f"--initial_training={self.conf.initial_training} " \
                       f"--neu_save_metrics={self.conf.neu_save_metrics}"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.conf.tf_device
            ret = subprocess.run(cmd_fn + " " + script_fn + " " + args_str, shell=True, env=env)
            if ret.returncode != 0:
                raise Exception("RREA did not run successfully.")

        # train simi to prob model
        simi_mtx = self.predict_simi()
        simi2prob_model = SimiToProbModule(conf=self.conf)
        simi2prob_model.train_model(simi_mtx)

    def train_model_with_observed_n_latent_labels(self):

        if self.conf.py_exe_fn is None:
            runner = Runner(self.conf.data_dir, self.conf.output_dir, enhanced=True,
                            max_train_epoch=self.conf.max_train_epoch,
                            max_continue_epoch=self.conf.max_continue_epoch
                            )
            runner.restore_model(self.conf.restore_from_dir)
            # runner.train()
            runner.continue_training()
            runner.save(save_metrics=self.conf.neu_save_metrics)
        else:
            cmd_fn = self.conf.py_exe_fn
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            script_fn = os.path.join(cur_dir, "Dual_AMN/runner.py")
            args_str = f"--data_dir={self.conf.data_dir} --output_dir={self.conf.output_dir} " \
                       f"--max_train_epoch={self.conf.max_train_epoch} --max_continue_epoch={self.conf.max_continue_epoch} " \
                       f"--initial_training={self.conf.initial_training} " \
                       f"--neu_save_metrics={self.conf.neu_save_metrics} --enhanced=True --restore_from_dir={self.conf.restore_from_dir}"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.conf.tf_device
            ret = subprocess.run(cmd_fn + " " + script_fn + " " + args_str, shell=True, env=env)
            if ret.returncode != 0:
                raise Exception("RREA did not run successfully")

        # train simi to prob model
        simi_mtx = self.predict_simi()
        simi2prob_model = SimiToProbModule(conf=self.conf)
        simi2prob_model.restore_from(self.conf.restore_from_dir)
        simi2prob_model.train_model(simi_mtx)


    def predict(self):
        simi_mtx = self.predict_simi()
        prob_mtx = self.convert_simi_to_probs7(simi_mtx)
        prob_mtx = torch.tensor(prob_mtx, device=torch.device("cuda:0"))
        del simi_mtx
        return prob_mtx

    def predict_simi(self):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        # simi_mtx = sim_handler(ent1_embs, ent2_embs, k=10, nums_threads=100)

        evaluator = Evaluator()
        simi_mtx = evaluator.csls_sim(ent1_embs, ent2_embs, k=10)

        return simi_mtx

    def get_embeddings(self):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]
        return ent1_embs, ent2_embs

    def get_pred_alignment(self):
        # emea_data = EMEAData(self.conf.data_dir, self.conf.data_name)
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json")) as file:
            obj = json.loads(file.read())
            pred_alignment = obj["pred_alignment_csls"]
        # kg1_old2new_ent_map, kg2_old2new_ent_map = emea_data.old2new_entity_id_map()
        # pred_alignment = [(kg1_old2new_ent_map[ent1], kg2_old2new_ent_map[ent2]) for ent1, ent2 in pred_alignment]
        return pred_alignment

    def enhance_latent_labels_with_jointdistr(self, improved_candi_probs: torch.Tensor, candi_mtx):
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        # idx_range = np.arange(improved_candi_probs.shape[1])
        sampling_num = 1
        sampling_ent2_list = []
        for ent1, ent2 in train_alignment:
            sampling_ent2_list.append([ent2]*sampling_num)
        with torch.no_grad():
            for ent1, _ in test_alignment:
                probs = improved_candi_probs[ent1]
                # tmp_idx = np.argmax(probs, axis=-1)
                tmp_idx = torch.argmax(probs, dim=-1, keepdim=False).cpu().numpy()
                ent2_arr = np.array([tmp_idx])
                sampling_ent2_list.append(ent2_arr)
        ent1_arr = np.concatenate([np.array(train_alignment)[:, 0], np.array(test_alignment)[:, 0]])
        ent2_mtx = np.stack(sampling_ent2_list, axis=0)
        new_alignment = np.stack([ent1_arr, ent2_mtx[:, 0]], axis=-1).tolist()
        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in new_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")

    def evaluate(self):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        eval_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))

        evaluator = Evaluator()
        cos_test_metrics, cos_test_alignment = evaluator.evaluate_cosine(ent1_embs, ent2_embs, eval_alignment)
        csls_test_metrics, csls_test_alignment = evaluator.evaluate_csls(ent1_embs, ent2_embs, eval_alignment)
        csls_all_alignment = evaluator.predict_alignment(ent1_embs, ent2_embs)

        metrics_obj = {"metrics_csls": csls_test_metrics, "metrics_cos": cos_test_metrics}
        print("csls metrics: ", csls_test_metrics)
        pred_alignment_obj = {"pred_alignment_csls": csls_test_alignment.tolist(),
                              "pred_alignment_cos": cos_test_alignment.tolist(),
                              "all_pred_alignment_csls": csls_all_alignment.tolist()
                              }
        with open(os.path.join(self.conf.output_dir, "metrics.json"), "w+") as file:
            file.write(json.dumps(metrics_obj))
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json"), "w+") as file:
            file.write(json.dumps(pred_alignment_obj))
        with open(os.path.join(self.conf.output_dir, "eval_metrics.json"), "w+") as file:
            file.write(json.dumps(csls_test_metrics))



