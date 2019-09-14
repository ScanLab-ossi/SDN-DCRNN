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

## Dataset Preparation
We assume the data format used is a folder of an experiment simulated using
[SDNSandbox](https://www.github.com/yossisolomon/SDNSandbox "SDNSanbox's Github page") - an SDN testbed for researchers.

The main inputs from the experiment folder are:
* The sFlow monitoring datagrams HD5 file `sflow-datagrams.hd5`
* The topology csv file `<topo_name>.graphml-topo.csv`
* The file listing the interfaces detected in the experiment `intfs-list`

## Running SDN-DCRNN
Using the following command SDN-DCRNN will analyze the experiment

`EXP_DIR=<EXP_DIR> run_sdn_dcrnn.sh`

The `EXP_DIR` environment variable will be used to give the folder containing the input files listed above.

#### Multi-Experiment Run
We have a helper script to analyze multiple datasets:

`EXP_DATA_PATH=<DIRECTORY_INCLUDING_DATASET_DIRECTORIES> run_sdn_dcrnn_multi.sh`
### Analysis Stages
The analysis stages performed will be:

* generate_training_data + HD5 ==> data npz
    * The generated train/val/test dataset will be saved at `$EXP_DIR/{train,val,test}.npz`.
* gen_adj_mx + csv + intfs-list ==> adjacency matrix pickle file
* gen_config + paths + ports num ==> config file
* dcrnn_train + config file ==> model
* run_demo + config file ==> new predictions
* plot_predictions + predictions ==> plots

#### Troubleshooting
During training there is a chance that the training loss will explode (become too high and overflow).

A workaround we used is to lower the used measurement unit (e.g. Mbps instead of Kbps).

Another solution can be to decrease the learning rate earlier in the learning rate schedule. 

## Citation
If you find this repository useful in your research, please cite the following papers:
####TODO: Add new SDN related citation
```
@inproceedings{}
```

####The original paper
```
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}
```
