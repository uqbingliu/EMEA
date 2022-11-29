# -*- coding: utf-8 -*-

from emea.conf import Config
from emea.emea_framework import EMEAFramework
import os
from emea.jointdistr_paris import ParisJointDistri
import argparse
from emea.neural_openea import OpenEAModule


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--data_name', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--res_dir', type=str)
parser.add_argument('--topK', type=int)
parser.add_argument('--em_iteration_num', type=int, default=10)
# parser.add_argument('--device', default="cuda:0", type=str)
# parser.add_argument('--second_device', default="cuda:1", type=str)
parser.add_argument('--tf_device', default="/gpu:0", type=str)
parser.add_argument('--initial_training', default="supervised", type=str)
parser.add_argument('--max_train_epoch', default=1200, type=int)
parser.add_argument('--max_continue_epoch', default=200, type=int)
# parser.add_argument('--eval_freq', default=50, type=int)
parser.add_argument('--neu_save_metrics', default=0, type=int)
parser.add_argument('--py_exe_fn', default=None, type=str)
parser.add_argument('--openea_arg_fn', type=str)
parser.add_argument('--openea_script_fn', type=str)
args = parser.parse_args()


conf = Config()
conf.update(vars(args))


if not os.path.exists(conf.output_dir):
    os.makedirs(conf.output_dir)

neural_ea_model = OpenEAModule(conf)
joint_distr_model = ParisJointDistri(conf)
em_framework = EMEAFramework(conf=conf, jointdistr_model=joint_distr_model, neural_ea_model=neural_ea_model)

em_framework.run_EM()
em_framework.evaluate()



