# Guiding Neural Entity Alignment with Compatibility

This repo contains the source code of paper "Guiding Neural Entity Alignment with Compatibility", which has been accepted at EMNLP 2022.

Download the used data from this [Dropbox directory](https://www.dropbox.com/sh/ff6fr22e636lby8/AACQYSPOkX3Oy7NUNk8L2dgqa?dl=0).
Decompress it and put it under `emea_code/` as shown in the folder structure below.

## Structure of Folders

```
emea_code/
  - datasets/
  - emea/
  - OpenEA/
  - scripts/
  - environment.yml
  - README.md
```

After you run EMEA, there will be a `output/` folder which stores the evaluation results.

## Device
Our experiments are run on one GPU server which is configured with 3 NVIDIA GeForce GTX 2080Ti GPUs and Ubuntu 20.04 OS.
We suggest you use at least two GPUs in case of Out-Of-Memeory Issue.


## Install Conda Environment
`cd` to the project directory first. Then, run the following command to install the major environment packages.
```shell
conda env create -f environment.yml
```

With the installed environment, you can run EMEA for RREA and Dual-AMN, which are SOTA neural EA models.
If you also want to run EMEA for AliNet, IPTransE, which are used to verify the generality of EMEA, please also install the following packages with `pip`:
```shell
pip install igraph
pip install python-Levenshtein
pip install gensim
pip install dataclasses
```

## Run Scripts
* Go to the scripts folder via `cd scripts/` firstly.
* Before running the any script, you can modify the `data_name`, `train_percent`, `initial_training` and other settings in the script according to your need.
* Settings about used GPU are: `CUDA_VISIBLE_DEVICES="0"` and `tf_device="1"`. The former one is for EMEA, while the other one is for the neural EA model.
* Starting with the default settings is a good option.
* **The evaluation results, including metrics of every step, are saved under the `output/` directory.**

Below are the scripts for different purposes:

* Run `sh run_emea_w_rrea.sh` to reproduce results of EMEA on the three 15K datasets (i.e. zh_en, ja_en, fr_en) shown in Table 1.
* Run `sh run_emea_w_rrea_100k.sh` to reproduce results of EMEA on the two 100K datasets (i.e. dbp_wd, dbp_yg) shown in Table 1. The source code RREA cannot run on GPU. So I run it on CPU and very long time would be taken.
* Run `sh run_emea_w_rrea_semi.sh` to reproduce results of semi-supervised RREA on the three 15K datasets shown in Table 2.
* To reproduce the generality of EMEA shown in Fig. 4, run `run_emea_w_dual_amn.sh`, `run_emea_w_alinet.sh`, `run_emea_w_iptranse.sh`.
We suggest you try Dual-AMN model first because AliNet and IPTransE are relatively slow.
* Run `sh run_emea_avoidconf.sh` to reproduce the results of _AvoidConf_ rule shown in Figure 6.


## Acknowledgement
We used the source codes of [RREA](https://github.com/MaoXinn/RREA), [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [OpenEA](https://github.com/nju-websoft/OpenEA).
