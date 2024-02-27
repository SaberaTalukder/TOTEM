import argparse
import numpy as np
import os


def create_full_data(args, type='train'):
    if type == 'train':
        weather = np.load(args.root_path + '/weather/train_data_x.npy')
        electricity = np.load(args.root_path + '/electricity/train_data_x.npy')
        traffic = np.load(args.root_path + '/traffic/train_data_x.npy')
        ETTm1 = np.load(args.root_path + '/ETTm1/train_data_x.npy')
        ETTm2 = np.load(args.root_path + '/ETTm2/train_data_x.npy')
        ETTh1 = np.load(args.root_path + '/ETTh1/train_data_x.npy')
        ETTh2 = np.load(args.root_path + '/ETTh2/train_data_x.npy')

    elif type == 'val':
        weather = np.load(args.root_path + '/weather/val_data_x.npy')
        electricity = np.load(args.root_path + '/electricity/val_data_x.npy')
        traffic = np.load(args.root_path + '/traffic/val_data_x.npy')
        ETTm1 = np.load(args.root_path + '/ETTm1/val_data_x.npy')
        ETTm2 = np.load(args.root_path + '/ETTm2/val_data_x.npy')
        ETTh1 = np.load(args.root_path + '/ETTh1/val_data_x.npy')
        ETTh2 = np.load(args.root_path + '/ETTh2/val_data_x.npy')

    elif type == 'test':
        weather = np.load(args.root_path + '/weather/test_data_x.npy')
        electricity = np.load(args.root_path + '/electricity/test_data_x.npy')
        traffic = np.load(args.root_path + '/traffic/test_data_x.npy')
        ETTm1 = np.load(args.root_path + '/ETTm1/test_data_x.npy')
        ETTm2 = np.load(args.root_path + '/ETTm2/test_data_x.npy')
        ETTh1 = np.load(args.root_path + '/ETTh1/test_data_x.npy')
        ETTh2 = np.load(args.root_path + '/ETTh2/test_data_x.npy')

    data = np.concatenate((weather, electricity, traffic, ETTm1, ETTm2, ETTh1, ETTh2), axis=0)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if type == 'train':
        np.save(args.save_path + 'train_data_x.npy', data)
    elif type == 'val':
        np.save(args.save_path + 'val_data_x.npy', data)
    elif type == 'test':
        np.save(args.save_path + 'test_data_x.npy', data)


def main(args):
    create_full_data(args, 'train')
    create_full_data(args, 'val')
    create_full_data(args, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--save_path', type=str, default='./data/ETT/', help='where to save the data')
    args = parser.parse_args()
    main(args)