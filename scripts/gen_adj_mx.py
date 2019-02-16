from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle
from networkx import Graph, algorithms
from collections import namedtuple

Link = namedtuple('Link', 'From, To, Latency_in_ms')


def get_adjacency_matrix(links, switches, normalized_k):
    """
    :param links: data frame with three columns: [from, to, distance].
    :param switches: list of switches.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    # Builds sensor id to index map.
    switch_to_ind = {}
    for i, sensor_id in enumerate(switches):
        switch_to_ind[sensor_id] = i
    # Graph is not directed as the links are assumed to be symmetric
    network = Graph()
    [network.add_node(switch) for switch in switches]
    for link in links.values:
        network.add_edge(link[0], link[1], latency_in_ms=link[2])

    dist_mx = np.zeros((len(switches), len(switches)), dtype=np.float32)
    dist_mx[:] = np.inf
    # Fills cells in the matrix with latencies
    latencies = dict(algorithms.shortest_path_length(network, weight='latency_in_ms'))
    for i in switch_to_ind.keys():
        for j in switch_to_ind.keys():
            if j not in latencies[i]:
                print('jumping i={0} j={1}'.format(i, j))
                continue
            dist_mx[switch_to_ind[i]][switch_to_ind[j]] = latencies[i][j]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    print(adj_mx)
    return adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--links_csv', type=str, default='data/sensor_graph/Bellcanada.graphml.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='data/sensor_graph/Bellcanada.adj_mat.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    links_df = pd.read_csv(args.links_csv, dtype={'From': 'str', 'To': 'str'})
    switches = set([link[0] for link in links_df.values] + [link[1] for link in links_df.values])
    # make the result reproducible by making it sorted
    switches = sorted(switches)
    adj_mx = get_adjacency_matrix(links_df, switches, args.normalized_k)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump(adj_mx, f, protocol=2)
