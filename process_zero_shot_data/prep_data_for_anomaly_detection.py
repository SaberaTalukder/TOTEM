import pdb
import numpy as np
import torch
import os
import argparse


def mask_me(mask_ratio, revin, notrevin):
    # these are our anomalies
    # random mask
    B, T = revin.shape
    mask = torch.zeros((B, T))  # zeros are normal, ones are anomalies

    num_set_to_ones = int(B*T*mask_ratio)
    indices = np.random.choice(B*T, num_set_to_ones, replace=False)

    mask = mask.contiguous()
    mask.flatten()[indices] = 1

    labels = mask
    revin = revin.masked_fill(mask == 1, 10).unsqueeze(-1)  # take the anomaly and set it
    notrevin = notrevin.masked_fill(mask == 1, 10).unsqueeze(-1)   # take the anomaly and set it

    return labels, revin, notrevin


def main(args):
    path = args.base_path
    save_path = args.save_path

    test_revin = torch.tensor(np.load(path + 'test_revin_x.npy', allow_pickle=True))
    test_notrevin = torch.tensor(np.load(path + 'test_notrevin_x.npy', allow_pickle=True))

    # split train and test in 1/2 because anomaly detection needs a train set to calculate a threshold
    cutoff = test_revin.shape[0]//2

    train_revin = test_revin[0:cutoff,:]
    train_notrevin = test_notrevin[0:cutoff, :]

    test_revin = test_revin[cutoff:, :]
    test_notrevin = test_notrevin[cutoff:, :]

    if test_revin.shape != test_notrevin.shape:
        pdb.set_trace()

    mask_ratio = 0.02  # 2% of the data

    test_labels, test_revin_masked, test_notrevin_masked = mask_me(mask_ratio, test_revin, test_notrevin)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_revin = train_revin.unsqueeze(-1)
    train_notrevin = train_notrevin.unsqueeze(-1)

    np.save(save_path + 'test_revin_masked.npy', test_revin_masked)
    np.save(save_path + 'test_notrevin_masked.npy', test_notrevin_masked)
    np.save(save_path + 'test_labels_totem_shaped.npy', test_labels)
    np.save(save_path + 'train_revin.npy', train_revin)
    np.save(save_path + 'train_notrevin.npy', train_notrevin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_path', type=str, default='', help='base path')
    parser.add_argument('--save_path', type=str, default='', help='place to save processed files')

    args = parser.parse_args()
    main(args)