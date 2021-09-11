from lib.utils import load_pickle, calculate_random_walk_matrix, calculate_normalized_laplacian, calculate_scaled_laplacian
import sys
import os
import argparse

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj-mx-file', type=str)
    args = parser.parse_args()
    adj_mx = load_pickle(args.adj_mx_file)
    print("random walk matrix:")
    print(calculate_random_walk_matrix(adj_mx=adj_mx))
    print("normalized_laplacian:")
    print(calculate_normalized_laplacian(adj=adj_mx))
    print("scaled_laplacian:")
    print(calculate_scaled_laplacian(adj_mx=adj_mx))