from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle
from networkx import Graph, algorithms
from collections import namedtuple

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
        network.add_edge(link[0], link[1], latency_in_ms=link[2])
    # Fills cells in the matrix with latencies
    switch_latencies = dict(algorithms.shortest_path_length(network, weight='latency_in_ms'))
    latencies = {}
    for i in ports.keys():
        curr_latencies = {}
        for j in ports.keys():
            if ports[i].Dst == ports[j].Dst:
                curr_latencies[j] = 0
            else:
                curr_latencies[j] = switch_latencies[ports[i].Dst][ports[j].Dst]
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
                        help='Path of the output file. Default - \"links-csv\".pkl')
    args = parser.parse_args()

    if args.output_pkl_filename == "":
        args.output_pkl_filename = args.links_csv + '.pkl'

    links_df = pd.read_csv(args.links_csv, dtype={'From': 'str', 'To': 'str'})
    switches = set([link[0] for link in links_df.values] + [link[1] for link in links_df.values])
    # make the result reproducible by making it sorted
    switches = sorted(switches)

    ports_map = get_ports_map(args.intfs_list, switches)

    latencies = get_latencies_map(links_df.values, switches, ports_map)

    adj_mx = get_adjacency_matrix(latencies, sorted(ports_map.keys()), args.normalized_k)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump(adj_mx, f, protocol=2)
