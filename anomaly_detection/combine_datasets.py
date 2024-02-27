import argparse
import pdb
import numpy as np
import torch
import os


def load_and_flatten(folder, args):
    train = np.load(args.root_path + '/' + folder + '/revin_data/train.npy', allow_pickle=True)
    test = np.load(args.root_path + '/' + folder + '/revin_data/test.npy', allow_pickle=True)

    train = np.swapaxes(train, 1, 2)
    test = np.swapaxes(test, 1, 2)

    train = train.reshape(-1, train.shape[-1])
    test = test.reshape(-1, test.shape[-1])

    print(train.shape)
    print(test.shape)
    print('-----------')

    return train, test

def main(args):
    folders = ['MSL', 'PSM', 'SMAP', 'SMD', 'SWAT']
    num_sensors = [55, 25, 25, 38, 51]

    all_train = []
    all_test = []

    for i, folder in enumerate(folders):
        print(folder)
        train, test = load_and_flatten(folder, args)
        all_train.append(train)
        all_test.append(test)

    all_train = np.concatenate(all_train, axis=0)
    all_test = np.concatenate(all_test, axis=0)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    pdb.set_trace()

    np.save(args.save_path + 'train.npy', all_train, allow_pickle=True)
    np.save(args.save_path + 'test.npy', all_test, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        required=True,
                        help='path to the data')

    parser.add_argument('--save_path', type=str,
                        required=True,
                        help='path to save the data')

    parser.add_argument('--gpu', type=int, default=1, help='gpu')

    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.gpu))

    main(args)