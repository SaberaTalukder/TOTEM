import argparse
import pdb
import numpy as np
import torch
import os


def load_and_flatten(folder, args, num_sensor):
    test_labels = np.load(args.root_path + '/' + folder + '/test_labels_processed.npy', allow_pickle=True)
    test_labels_repeated = np.repeat(test_labels[:, :, np.newaxis], num_sensor, axis=2)
    test_labels_swapped = np.swapaxes(test_labels_repeated, 1, 2)
    test_labels_repeated_reshaped = test_labels_swapped.reshape(-1, test_labels_swapped.shape[-1])
    print(test_labels_repeated_reshaped.shape)
    return test_labels_repeated_reshaped


def main(args):
    folders = ['MSL', 'PSM', 'SMAP', 'SMD', 'SWAT']
    num_sensors = [55, 25, 25, 38, 51]
    all_test_labels = []

    for i, folder in enumerate(folders):
        print(folder)
        test_labels = load_and_flatten(folder, args, num_sensors[i])
        all_test_labels.append(test_labels)

    all_test_labels_arr = np.concatenate(all_test_labels, axis=0)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    pdb.set_trace()
    np.save(args.save_path + 'test_labels_processed.npy', all_test_labels_arr, allow_pickle=True)


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