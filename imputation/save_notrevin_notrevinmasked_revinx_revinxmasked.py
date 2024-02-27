import argparse
import numpy as np
import os
import pdb
import random
import torch

from data_provider.data_factory import data_provider
from layers.RevIN import RevIN


class ExtractData:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:' + str(self.args.gpu)
        self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def one_loop(self, loader):
        x_original = []
        x_in_revin_space = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x_original.append(np.array(batch_x))
            batch_x = batch_x.float().to(self.device)
            # data going into revin should have dim:[bs x seq_len x nvars]
            x_in_revin_space.append(np.array(self.revin_layer_x(batch_x, "norm").detach().cpu()))

        x_original_arr = np.concatenate(x_original, axis=0)
        x_in_revin_space_arr = np.concatenate(x_in_revin_space, axis=0)

        print(x_in_revin_space_arr.shape, x_original_arr.shape)
        return x_in_revin_space_arr, x_original_arr

    def extract_data(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        print('got loaders starting revin')

        # These have dimension [bs, ntime, nvars]
        x_train_in_revin_space_arr, x_train_original_arr = self.one_loop(train_loader)
        print('starting val')
        x_val_in_revin_space_arr, x_val_original_arr = self.one_loop(vali_loader)
        print('starting test')
        x_test_in_revin_space_arr, x_test_original_arr = self.one_loop(test_loader)

        print('Flattening Sensors Out')
        if self.args.seq_len != 96 and self.args.pred_len != 0 :
            pdb.set_trace()
        else:
            # These have dimension [bs x nvars, ntime]
            x_train_arr = np.swapaxes(x_train_in_revin_space_arr, 1,2).reshape((-1, self.args.seq_len))
            x_val_arr = np.swapaxes(x_val_in_revin_space_arr, 1, 2).reshape((-1, self.args.seq_len))
            x_test_arr = np.swapaxes(x_test_in_revin_space_arr, 1, 2).reshape((-1, self.args.seq_len))

            orig_x_train_arr = np.swapaxes(x_train_original_arr, 1, 2).reshape((-1, self.args.seq_len))
            orig_x_val_arr = np.swapaxes(x_val_original_arr, 1, 2).reshape((-1, self.args.seq_len))
            orig_x_test_arr = np.swapaxes(x_test_original_arr, 1, 2).reshape((-1, self.args.seq_len))

            print(x_train_arr.shape, x_val_arr.shape, x_test_arr.shape)
            print(orig_x_train_arr.shape, orig_x_val_arr.shape, orig_x_test_arr.shape)

        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        np.save(self.args.save_path + '/train_revin_x.npy', x_train_arr)
        np.save(self.args.save_path + '/val_revin_x.npy', x_val_arr)
        np.save(self.args.save_path + '/test_revin_x.npy', x_test_arr)

        np.save(self.args.save_path + '/train_notrevin_x.npy', orig_x_train_arr)
        np.save(self.args.save_path + '/val_notrevin_x.npy', orig_x_val_arr)
        np.save(self.args.save_path + '/test_notrevin_x.npy', orig_x_test_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Formers
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # Save Location
    parser.add_argument('--save_path', type=str, default=False, help='folder ending in / where we want to save the revin data to')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print(args.enc_in)
    print('CHECK make sure you have the right enc_in')
    print('Weather : 21')
    print('Electricity : 321')
    print('ETTm1, m2, h1, h2 : 7')

    if 'electricity' in args.data_path and args.data == 'custom' and args.enc_in == 321:
        pass
    elif 'weather' in args.data_path and args.data == 'custom' and args.enc_in == 21:
        pass
    elif 'ETT' in args.data and args.enc_in == 7:
        pass
    else:
        pdb.set_trace()

    print('Args in experiment:')
    print(args)

    Exp = ExtractData
    exp = Exp(args)  # set experiments
    exp.extract_data()
