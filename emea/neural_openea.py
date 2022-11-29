# -*- coding: utf-8 -*-


from emea.emea_framework import NeuralEAModule
from emea.conf import Config
import os
import subprocess
from emea.simi_to_prob import SimiToProbModule
import numpy as np
from emea.RREA.CSLS_torch import Evaluator
import torch
from emea.data import load_alignment
import json
from emea.data import update_openea_enhanced_train_data


class OpenEAModule(NeuralEAModule):
    def __init__(self, conf: Config):
        super(OpenEAModule, self).__init__(conf)

    # def prepare_data(self):
    #     pass

    def train_model_with_observed_labels(self):
        cmd_fn = self.conf.py_exe_fn
        script_fn = self.conf.openea_script_fn
        args_str = f"{self.conf.openea_arg_fn} {self.conf.data_dir}/openea_format/ partition {self.conf.output_dir}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.conf.tf_device
        env["PYTHONPATH"] = os.path.join(os.path.dirname(script_fn), "../src/") #+ ";" + env["PYTHONPATH"]
        ret = subprocess.run(cmd_fn + " " + script_fn + " " + args_str, shell=True, env=env)
        if ret.returncode != 0:
            raise Exception("AliNet did not run successfully.")

        # train simi to prob model
        simi_mtx = self.predict_simi()
        simi2prob_model = SimiToProbModule(conf=self.conf)
        simi2prob_model.train_model(simi_mtx)

    def train_model_with_observed_n_latent_labels(self):
        update_openea_enhanced_train_data(self.conf.data_dir)
        self.train_model_with_observed_labels()

    def predict_simi(self):
        ent1_embs, ent2_embs = self.get_embeddings()
        evaluator = Evaluator()
        simi_mtx = evaluator.csls_sim(ent1_embs, ent2_embs, k=10)
        return simi_mtx

    def predict(self):
        simi_mtx = self.predict_simi()
        prob_mtx = self.convert_simi_to_probs7(simi_mtx)
        prob_mtx = torch.tensor(prob_mtx, device=torch.device("cuda:0"))
        del simi_mtx
        return prob_mtx

    def get_embeddings(self):
        embs = np.load(os.path.join(self.conf.output_dir, "ent_embeds.npy"))
        with open(os.path.join(self.conf.output_dir, "kg1_ent_ids")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            tuples = [line.split() for line in lines]
            tuples = [(uri, int(id)) for uri, id in tuples]
            kg1_ent_uri2id_map = dict(tuples)
        with open(os.path.join(self.conf.output_dir, "kg2_ent_ids")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            tuples = [line.split() for line in lines]
            tuples = [(uri, int(id)) for uri, id in tuples]
            kg2_ent_uri2id_map = dict(tuples)
        with open(os.path.join(self.conf.data_dir, "ent_ids_1")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            uri_list = [line.split()[1] for line in lines]
            kg1_ent_id_list = [kg1_ent_uri2id_map[uri] for uri in uri_list]
        with open(os.path.join(self.conf.data_dir, "ent_ids_2")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            uri_list = [line.split()[1] for line in lines]
            kg2_ent_id_list = [kg2_ent_uri2id_map[uri] for uri in uri_list]

        ent1_ids = np.array(kg1_ent_id_list)
        ent2_ids = np.array(kg2_ent_id_list)
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]
        return ent1_embs, ent2_embs

    def get_pred_alignment(self):
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json")) as file:
            obj = json.loads(file.read())
            pred_alignment = obj["pred_alignment_csls"]
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
        ent1_embs, ent2_embs = self.get_embeddings()

        eval_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))

        evaluator = Evaluator()
        cos_test_metrics, cos_test_alignment = evaluator.evaluate_cosine(ent1_embs, ent2_embs, eval_alignment)
        csls_test_metrics, csls_test_alignment = evaluator.evaluate_csls(ent1_embs, ent2_embs, eval_alignment)

        metrics_obj = {"metrics_csls": csls_test_metrics, "metrics_cos": cos_test_metrics}
        print("csls metrics: ", csls_test_metrics)
        pred_alignment_obj = {"pred_alignment_csls": csls_test_alignment.tolist(),
                              "pred_alignment_cos": cos_test_alignment.tolist()}
        with open(os.path.join(self.conf.output_dir, "metrics.json"), "w+") as file:
            file.write(json.dumps(metrics_obj))
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json"), "w+") as file:
            file.write(json.dumps(pred_alignment_obj))
        with open(os.path.join(self.conf.output_dir, "eval_metrics.json"), "w+") as file:
            file.write(json.dumps(csls_test_metrics))


