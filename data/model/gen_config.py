import yaml
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-t", "--template_file", type=str, default="dcrnn_config_template.yaml"
)
parser.add_argument(
    "-o", "--output_prefix", type=str, required=True
)
parser.add_argument(
    "-d", "--dataset_dir", type=str, required=True, help="Dataset directory."
)
parser.add_argument(
    "-p", "--graph_adj_mx_pkl", type=str, required=True, help="PKL file representing th network adjacency matrix."
)
parser.add_argument(
    "-n", "--num_ports", type=int, required=True, help="Number of ports in network."
)
args = parser.parse_args()


with open(args.template_file, 'r') as stream:
    data = yaml.load(stream)

data['base_dir'] = args.dataset_dir
data['data']['dataset_dir'] = args.dataset_dir
data['data']['graph_pkl_filename'] = args.graph_adj_mx_pkl
data['model']['num_nodes'] = args.num_ports

with open(args.output_prefix + args.template_file, 'w') as yaml_file:
    yaml_file.write( yaml.dump(data, default_flow_style=False))