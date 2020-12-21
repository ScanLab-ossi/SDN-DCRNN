import yaml
import argparse

default_template = """---
base_dir: data/model
log_level: INFO
data:
  batch_size: 64
  dataset_dir: DATASET_DIR
  test_batch_size: 64
  val_batch_size: 64
  graph_pkl_filename: GRAPH_ADJ_MX_PKL

model:
  cl_decay_steps: 2000
  filter_type: dual_random_walk
  horizon: HORIZON
  input_dim: INPUT_DIM
  l1_decay: 0
  max_diffusion_step: 2
  num_nodes: NUM_PORTS
  num_rnn_layers: 2
  output_dim: 1
  rnn_units: 16
  seq_len: SEQ_LEN
  use_curriculum_learning: true

train:
  base_lr: 0.01
  dropout: 0
  epoch: 0
  epochs: 100
  epsilon: 1.0e-3
  global_step: 0
  lr_decay_ratio: 0.1
  max_grad_norm: 5
  max_to_keep: 100
  min_learning_rate: 2.0e-06
  optimizer: adam
  patience: 50
  steps: [20, 30, 40, 50]
  test_every_n_epochs: 10
"""

parser = argparse.ArgumentParser()

parser.add_argument(
    "-t", "--template_file", type=str
)
parser.add_argument(
    "-d", "--dataset_dir", type=str, required=True, help="Dataset directory."
)
parser.add_argument(
    "-p", "--graph_adj_mx_pkl", type=str, required=True, help="PKL file representing the network adjacency matrix."
)
parser.add_argument(
    "-n", "--num_ports", type=int, required=True, help="Number of ports in network."
)
parser.add_argument(
    "--horizon", type=int, required=True, help="Number of seconds to predict forward."
)
parser.add_argument(
    "--seq_len", type=int, required=True, help="Length of the sequence for input to the DCRNN algorithm."
)
parser.add_argument(
    "--input-dim", type=int, required=True, help="Dimensions for input to the DCRNN algorithm."
)
parser.add_argument(
    "--output-path", default="sdn-dcrnn-config.yaml", help="Name of output file."
)
args = parser.parse_args()

if args.template_file is not None:
    with open(args.template_file, 'r') as template:
        data = yaml.load(template)
else:
    data = yaml.load(default_template)

data['base_dir'] = args.dataset_dir
data['data']['dataset_dir'] = args.dataset_dir
data['data']['graph_pkl_filename'] = args.graph_adj_mx_pkl
data['model']['num_nodes'] = args.num_ports
data['model']['horizon'] = args.horizon
data['model']['seq_len'] = args.seq_len
data['model']['input_dim'] = args.input_dim

yaml_filename = args.output_file
print('Outputting yaml config to ' + yaml_filename)
with open(yaml_filename, 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)
