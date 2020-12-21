import argparse
import json
from typing import Set, Dict

import numpy as np
import pickle
from networkx import Graph, algorithms
from collections import namedtuple

Port = namedtuple('Port', 'Src, Dst')


def get_adjacency_matrix(latencies, ports, zero_under):
    """
    :param latencies: map of port to map of port latencies from key port
    :param ports: list of Port IDs.
    :param zero_under: Entries lower than this are set to zero for sparsity.
    :return:
    """
    # Builds sensor id to index map.
    port_to_ind = {}
    for i, port in enumerate(ports):
        port_to_ind[port] = i
    print('Using the following map of port (interface) ID to enumeration:')
    print(port_to_ind)

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
    adj_mx[adj_mx < zero_under] = 0
    print(adj_mx)
    return adj_mx


def get_latencies_map(links, switches: Set[str], ports: Dict[int, Port]):
    # Graph is not directed as the links are assumed to be symmetric
    network = Graph()
    [network.add_node(switch) for switch in switches]
    for link in links:
        network.add_edge(str(link['first_id']), str(link['second_id']),
                         latency_in_ms=float(link['mininet_latency'][:-2]))
    # Fills cells in the matrix with latencies
    switch_latencies = dict(algorithms.shortest_path_length(network, weight='latency_in_ms'))
    latencies = {}
    for i in ports.keys():
        curr_latencies = {}
        for j in ports.keys():
            curr_latencies[j] = 0 if ports[i].Dst == ports[j].Dst else switch_latencies[ports[i].Dst][ports[j].Dst]
        latencies[i] = curr_latencies
    return latencies


def get_ports_map(interfaces, switch_ids):
    ports = {}
    for interface in interfaces:
        if_parts = interface['name'].split('@')
        if len(if_parts) == 2:
            src_switch_name = if_parts[0].split('-')[0]
            src_switch_id = src_switch_name[1:]
            dst_switch_name = if_parts[1].split('-')[0]
            dst_switch_id = dst_switch_name[1:]
            if src_switch_id in switch_ids and dst_switch_id in switch_ids:
                ports[interface['num']] = Port(src_switch_id, dst_switch_id)
    return ports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network-data-json', required=True, type=str,
                        help='JSON file containing data about the network - '
                             'interfaces, switches and links between switches.')
    parser.add_argument('-z', '--zero-under', type=float, default=0.1,
                        help='Entries lower than this are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='adj_mx.pkl',
                        help='Path of the pickle output file.')
    parser.add_argument('--output_ports_num_filename', type=str, default='ports.num',
                        help='Path of the ports num output file.')
    args = parser.parse_args()

    with open(args.network_data_json) as network_data_file:
        network_data = json.load(network_data_file)
    print("Loaded the following network data:")
    print(network_data)
    switch_ids = set(str(sw['ID']) for sw in network_data['switches'].values())
    print("Switch IDs: " + ', '.join(switch_ids))
    ports_map = get_ports_map(network_data['interfaces'].values(), switch_ids)

    ports_num = str(len(ports_map))
    print("Amount of ports found: " + ports_num)
    with open(args.output_ports_num_filename, 'w') as f:
        f.write(ports_num)

    latencies = get_latencies_map(network_data['switch_links'], switch_ids, ports_map)

    adj_mx = get_adjacency_matrix(latencies, sorted(ports_map.keys()), args.zero_under)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump(adj_mx, f, protocol=2)
