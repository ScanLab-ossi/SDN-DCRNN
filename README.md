# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting - SDN Usecase

![Diffusion Convolutional Recurrent Neural Network](figures/model_architecture.jpg "Model Architecture")

This repository is a fork of the original design to suit the analysis of traffic load in Software Defined Networks (SDNs).

This is a TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network from the following paper: \
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926), ICLR 2018.


## Requirements
Dependencies can be found in the "requirements.txt" file.

They can be easily installed using the following command:
```bash
pip install -r requirements.txt
```
If this is an issue (to do with python package management), you can use a virtualenv python environment.
## Dataset Preparation
We assume the data format used is a folder of an experiment simulated using
[SDNSandbox](https://www.github.com/ScanLab-ossi/SDNSandbox "SDNSanbox's Github page") - an SDN testbed for researchers.

The main inputs from the experiment folder are:
* The sFlow monitoring datagrams HD5 file `sflow-datagrams.hd5`
* The topology csv file `<topo_name>.graphml-topo.csv`
* The file listing the interfaces detected in the experiment `intfs-list`

## Running SDN-DCRNN
Using the following command SDN-DCRNN will analyze the experiment

`EXP_DIR=<EXP_DIR> ./run_sdn_dcrnn.sh`

The `EXP_DIR` environment variable will be used to give the folder containing the input files listed above.

The variable `HORIZON` must be set to the amount of datapoints to forecast into the future

The variable `SEQ_LEN` must be set to the amount of datapoints to be used as input - using the same number as `HORIZON`
is usually a good way to start

The variable `PERIOD_CYCLE` will be used to decide whether to use [time in experiment modulo period cycle length] as a second input dimension -
this will usually majorly increase the accuracy of the predictions
#### Multi-Experiment Run
We have a helper script to analyze multiple datasets:

`EXP_DATA_PATH=<DIRECTORY_INCLUDING_DATASET_DIRECTORIES> ./run_sdn_dcrnn_multi.sh`

The variable `EXP_DATA_PATH` must be set to the storage path containing data directories for analysis

The variable `PARALLEL` can be used to influence parallelization of the analysis - otherwise a default will be used

Other variables from the regular single experiment run can be used as well, except `EXP_DIR`
### Analysis Stages
The analysis stages performed will be:

* generate_training_data + HD5 ==> data npz
    * The generated train/val/test dataset will be saved at `$EXP_DIR/{train,val,test}.npz`.
* gen_adj_mx + csv + intfs-list ==> adjacency matrix pickle file
* gen_config + paths + ports num ==> config file
* dcrnn_train + config file ==> model
* run_demo + config file ==> new predictions
* plot_predictions + predictions ==> plots

### Normalized Error Rates Comparison
In order to compare different results with plots, the following helper scripts were created:
`compare_sdn_dcrnn_dataset_results.sh`

Variables `EXP_DATA_PATH`, `HORIZON` must be used as above to indicate the required way you wish to run the comparison

### Baseline Algorithms Calculation
In order to compare to 3 baseline algorithms (static, historical average, VAR),  with error rates RMSE, MAE, MAPE and checks at horizons 5, 10, 20, 30 run the following:

`python -m scripts.eval_baseline_methods --traffic_reading_filename <path_to_sdnsandbox.hd5> --horizons 5 10 20 30`

#### Troubleshooting
During training there is a chance that the training loss will explode (become too high and overflow).

A workaround we used is to lower the used measurement unit (e.g. Mbps instead of Kbps).

Another solution can be to decrease the learning rate earlier in the learning rate schedule. 

## Our Datasets
Our datasets were generated using SDNSandbox for the following network topologies:
* Agis - national USA network
* Bellsouth - regional southern USA
* Tw - national USA network
* Deltacom - regional eastern USA seaboard

We created one for the identical load configuration and another for the shifted load one.
All configurations and logs are included in the dataset.

* [Identical load dataset](https://www.kaggle.com/datasets/yossisolomon/sdnsandbox-identical-load)
* [Shifted load dataset](https://www.kaggle.com/datasets/yossisolomon/sdnsandbox-shifted-load)

## Citation
If you find this repository useful in your research, please cite the following papers:

```
@article{SOLOMON2022109446,
title = {SDNSandbox — Enabling learning-based innovation in provider networks},
journal = {Computer Networks},
volume = {219},
pages = {109446},
year = {2022},
issn = {1389-1286},
doi = {https://doi.org/10.1016/j.comnet.2022.109446},
url = {https://www.sciencedirect.com/science/article/pii/S1389128622004807},
author = {Yossi Solomon and Osnat Mokryn and Tsvi Kuflik}}
```

### The original paper
```
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}
```
