# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting - SDN Usecase

![Diffusion Convolutional Recurrent Neural Network](figures/model_architecture.jpg "Model Architecture")

This is a TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network in the following paper: \
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926), ICLR 2018.

This repository is a fork of the original design to suit the analysis of Software Defined Networks (SDNs).
## Requirements
- scipy>=0.19.0
- numpy>=1.12.1
- pandas>=0.19.2
- tensorflow>=1.3.0
- pyaml


Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
### Sample Transformation
TODO: explain scripts and SDNSandbox
```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```
The generated train/val/test dataset will be saved at `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.

### Graph Construction
 As the currently implementation is based on pre-calculated road network distances between sensors, it currently only
 supports sensor ids in Los Angeles (see `data/sensor_graph/sensor_info_201206.csv`).

```bash
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```

## Model Training
```bash
# METR-LA
python dcrnn_train.py --config_filename=data/model/dcrnn_la.yaml

# PEMS-BAY
python dcrnn_train.py --config_filename=data/model/dcrnn_bay.yaml
```
Each epoch takes about 5min or 10 min on a single GTX 1080 Ti for METR-LA or PEMS-BAY respectively. 

There is a chance that the training loss will explode, the temporary workaround is to restart from the last saved model before the explosion, or to decrease the learning rate earlier in the learning rate schedule. 

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}
```
TODO: Add new citation