from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle
from networkx import Graph, algorithms
from collections import namedtuple

Link = namedtuple('Link', 'From_ID, From_Name, To_ID, To_Name, Latency_in_ms')
Port = namedtuple('Port', 'Src, Dst')


def get_adjacency_matrix(latencies, ports, normalized_k):
    """
    :param latencies: map of port to map of port latencies from key port
    :param ports: list of Ports.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    # Builds sensor id to index map.
    port_to_ind = {}
    for i, port in enumerate(ports):
        port_to_ind[port] = i

    dist_mx = np.zeros((len(ports), len(ports)), dtype=np.float32)
    dist_mx[:] = np.inf

    for i in port_to_ind.keys():
        for j in port_to_ind.keys():
            if j not in latencies[i]:
                print('jumping i={0} j={1}'.format(i, j))
                continue
            dist_mx[port_to_ind[i]][port_to_ind[j]] = latencies[i][j]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    print(adj_mx)
    return adj_mx


def get_latencies_map(links, switches, ports):
    # Graph is not directed as the links are assumed to be symmetric
    network = Graph()
    [network.add_node(switch) for switch in switches]
    for link in links:
        network.add_edge('s'+str(link.From_ID), 's'+str(link.To_ID), latency_in_ms=link.Latency_in_ms)
    # Fills cells in the matrix with latencies
    switch_latencies = dict(algorithms.shortest_path_length(network, weight='latency_in_ms'))
    latencies = {}
    for i in ports.keys():
        curr_latencies = {}
        for j in ports.keys():
            if ports[i].Dst == ports[j].Dst:
                curr_latencies[j] = 0
            else:
                port_i_dst = ports[i].Dst
                port_j_dst = ports[j].Dst
                curr_latencies[j] = switch_latencies[port_i_dst][port_j_dst]
        latencies[i] = curr_latencies

    return latencies


def get_ports_map(intfs_list_filename, switch_names_set):
    ports = {}
    with open(intfs_list_filename) as intfs_list:
        for row in intfs_list:
            if_num, if_name = row.split(': ')
            if_parts = if_name.split('@')
            if len(if_parts) == 2:
                src_switch_name = if_parts[0].split('-')[0]
                dst_switch_name = if_parts[1].split('-')[0]
                if src_switch_name in switch_names_set and dst_switch_name in switch_names_set:
                    ports[if_num] = Port(src_switch_name, dst_switch_name)
    return ports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--links_csv', required=True, type=str,
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument("-i", "--intfs_list", required=True,
                        help="Input file listing the interfaces during the experiment")
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='',
                        help='Path of the pickle output file. Default - \"links-csv\".pkl')
    parser.add_argument('--output_ports_num_filename', type=str, default='',
                        help='Path of the ports num output file. Default - \"links-csv\".port.num')
    args = parser.parse_args()

    if args.output_pkl_filename == "":
        args.output_pkl_filename = args.links_csv + '.pkl'

    if args.output_ports_num_filename == "":
        args.output_ports_num_filename = args.links_csv + '.port.num'

    links_df = pd.read_csv(args.links_csv)
    links = [Link(*link[1:]) for link in links_df.itertuples()]
    switches = set(['s'+str(link.To_ID) for link in links] + ['s'+str(link.From_ID) for link in links])
    # make the result reproducible by making it sorted
    switches = sorted(switches)
    print("Switches found: " + str(switches))
    ports_map = get_ports_map(args.intfs_list, switches)

    ports_num = str(len(ports_map))
    print("Amount of ports found: " + ports_num)
    with open(args.output_ports_num_filename, 'w') as f:
        f.write(ports_num)

    latencies = get_latencies_map(links, switches, ports_map)

    adj_mx = get_adjacency_matrix(latencies, sorted(ports_map.keys()), args.normalized_k)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump(adj_mx, f, protocol=2)
