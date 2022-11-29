# -*- coding: utf-8 -*-

from emea.conf import Config
import os
from emea.neural_rrea import RREAModule
from emea.jointdistr_avoidconf import AvoidConfJointDistri
from emea.emea_framework import EMEAFramework
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--data_name', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--res_dir', required=True, type=str)
parser.add_argument('--topK', required=True, type=int)
parser.add_argument('--conflict_checker_neigh_num', required=True, type=int)
parser.add_argument('--em_iteration_num', default=10, type=int)
parser.add_argument('--tf_device', default="/gpu:0", type=str)
parser.add_argument('--initial_training', default="supervised", type=str)
parser.add_argument('--max_train_epoch', default=1200, type=int)
parser.add_argument('--max_continue_epoch', default=200, type=int)
parser.add_argument('--py_exe_fn', default=None, type=str)

args = parser.parse_args()


conf = Config()
conf.update(vars(args))
# conf.device = "cuda:1"

if not os.path.exists(conf.output_dir):
    os.makedirs(conf.output_dir)


neural_ea_model = RREAModule(conf)
joint_distr_model = AvoidConfJointDistri(conf)
em_framework = EMEAFramework(conf=conf, jointdistr_model=joint_distr_model, neural_ea_model=neural_ea_model)
em_framework.run_EM()
em_framework.evaluate()



