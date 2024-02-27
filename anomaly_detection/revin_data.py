import argparse
import numpy as np
import os
import pdb
import torch
from layers.RevIN import RevIN


class ExtractData:
    def __init__(self, data, device, num_features):
        self.revin_layer = RevIN(num_features=num_features, affine=False, subtract_last=False)
        self.data = data
        self.device = device

    def one_loop(self, x):
        x_in_revin_space = []
        # data should have dimension batch, time, sensors (features)
        batch_x = torch.tensor(x)
        batch_x = batch_x.float().to(self.device)

        # data going into revin should have dim:[bs x seq_len x nvars]
        x_in_revin_space.append(np.array(self.revin_layer(batch_x, "norm").detach().cpu()))
        x_in_revin_space_arr = np.concatenate(x_in_revin_space, axis=0)

        print(x_in_revin_space_arr.shape)
        return x_in_revin_space_arr

    def extract_data(self):
        print('about to start revin')
        # These have dimension [bs, ntime, nvars]
        data_in_revin_space_arr = self.one_loop(self.data)
        return data_in_revin_space_arr


def do_data(data, device, num_feature):
    # data should have dimension: [bs x seq_len x nvars] --> so need to swap axis 1 & 2
    Exp = ExtractData
    exp = Exp(data, device, num_feature)  # set experiments
    return exp.extract_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        required=True,
                        help='path to the data')

    parser.add_argument('--save_path', type=str,
                        required=True,
                        help='path to save the data')

    parser.add_argument('--seq_len', type=int,
                        required=True,
                        help='window size to reconstruct')

    parser.add_argument('--num_vars', type=int,
                        required=False,
                        help='number of sensors')

    parser.add_argument('--gpu', type=int, default=1, help='gpu')

    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.gpu))

    base_path = args.root_path
    base_save_path = args.save_path
    num_features = args.num_vars

    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    else:
        print('The data path already exists')
        pdb.set_trace()

    # do train first
    data_train = np.load(base_path + "train_data_processed.npy", allow_pickle=True)
    print('initial train shape:', data_train.shape)
    data_train_revined = do_data(data_train, device, num_feature=num_features)

    data_test = np.load(base_path + "test_data_processed.npy", allow_pickle=True)
    print('initial test shape:', data_test.shape)
    data_test_revined = do_data(data_test, device, num_feature=num_features)

    print(data_train_revined.shape)
    print(data_test_revined.shape)

    if data_train_revined.shape[1] != args.seq_len or data_train_revined.shape[2] != args.num_vars:
        print('Train has shape problem')
        pdb.set_trace()

    if data_test_revined.shape[1] != args.seq_len or data_test_revined.shape[2] != args.num_vars:
        print('Test has shape problem')
        pdb.set_trace()

    np.save(base_save_path + '/train.npy', data_train_revined, allow_pickle=True)
    np.save(base_save_path + '/test.npy', data_test_revined, allow_pickle=True)

    print('FINISHED STEP 2')
    print('---------------')
