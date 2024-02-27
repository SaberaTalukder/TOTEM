import argparse
import pdb
import os
import numpy as np
import data_provider.data_factory as data_factory


def save_data(data_set, args, save_labels=False):
    all_data = []
    all_labels = []
    for i, (example, label) in enumerate(data_set):
        if example.shape != (args.seq_len, args.num_vars):  # do this becuase their loaders have the train set come with the test_labels (which are longer)
            break

        all_data.append(example.reshape(1, example.shape[0], example.shape[-1]))
        if save_labels:
            all_labels.append(label.reshape(1, label.shape[0]))

    if save_labels:
        return np.concatenate(all_data, axis=0), np.concatenate(all_labels, axis=0)
    else:
        return np.concatenate(all_data, axis=0), None


def process_data(args):
    train_data_set, train_data_loader = data_factory.data_provider(args, 'train')
    all_train, _ = save_data(train_data_set, args, save_labels=False)

    test_data_set, test_data_loader = data_factory.data_provider(args, 'test')
    all_test, all_test_labels = save_data(test_data_set, args, save_labels=True)

    print(all_train.shape)
    print(all_test.shape)
    print(all_test_labels.shape)

    if all_train.shape[1] != args.seq_len or all_train.shape[2] != args.num_vars:
        print('train shape off')
        pdb.set_trace()

    if all_test.shape[1] != args.seq_len or all_test.shape[2] != args.num_vars:
        print('test shape off')
        pdb.set_trace()

    if all_test_labels.shape[1] != args.seq_len:
        print('test labels shape off')
        pdb.set_trace()

    if all_test.shape[0] != all_test_labels.shape[0]:
        print('something funky with test sizes ')
        pdb.set_trace()

    pdb.set_trace()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    np.save(args.save_path + 'train_data_processed.npy', all_train, allow_pickle=True)
    np.save(args.save_path + 'test_data_processed.npy', all_test, allow_pickle=True)
    np.save(args.save_path + 'test_labels_processed.npy', all_test_labels, allow_pickle=True)

    print('FINISHED STEP 1')
    print('---------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        required=False, default='',
                        help='dataset name')
    parser.add_argument('--batch_size', type=int,
                        required=True,
                        help='batchsize')

    parser.add_argument('--task_name', type=str,
                        required=True,
                        help='name of the task')

    parser.add_argument('--root_path', type=str,
                        required=True,
                        help='path to the data')

    parser.add_argument('--save_path', type=str,
                        required=True,
                        help='path to save the data')

    parser.add_argument('--seq_len', type=int,
                        required=True,
                        help='window size to reconstruct')

    parser.add_argument('--num_workers', type=int,
                        required=False, default=10,
                        help='number of workers')

    parser.add_argument('--num_vars', type=int,
                        required=False,
                        help='number of sensors')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()

    if args.data == 'MSL' or args.data == 'PSM' or args.data == 'SMAP' or args.data == 'SMD' or args.data == 'SWAT':
        process_data(args)