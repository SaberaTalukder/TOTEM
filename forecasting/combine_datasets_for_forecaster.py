import argparse
import numpy as np
import os
import pdb


def check_and_save_codebook(args, data_folders):
    for df in data_folders:

        if not os.path.exists(args.save_path + df):
            os.makedirs(args.save_path + df)

        cbw = np.load(args.root_path + '/weather/' + df + '/codebook.npy')
        cbe = np.load(args.root_path + '/electricity/' + df + '/codebook.npy')
        cbt = np.load(args.root_path + '/traffic/' + df + '/codebook.npy')
        cbm1 = np.load(args.root_path + '/ETTm1/' + df + '/codebook.npy')
        cbm2 = np.load(args.root_path + '/ETTm2/' + df + '/codebook.npy')
        cbh1 = np.load(args.root_path + '/ETTh1/' + df + '/codebook.npy')
        cbh2 = np.load(args.root_path + '/ETTh2/' + df + '/codebook.npy')

        if np.array_equal(cbw, cbe) == True:
            if np.array_equal(cbw, cbt) == True:
                if np.array_equal(cbw, cbm1) == True:
                    if np.array_equal(cbw, cbm2) == True:
                        if np.array_equal(cbw, cbh1) == True:
                            if np.array_equal(cbw, cbh2) == True:
                                np.save(args.save_path + df + '/codebook.npy', cbw)


def concatenate(args, data_folders, name=''):
    for df in data_folders:

        w = np.load(args.root_path + '/weather/' + df + '/' + name + '.npy')
        e = np.load(args.root_path + '/electricity/' + df + '/' + name + '.npy')
        t = np.load(args.root_path + '/traffic/' + df + '/' + name + '.npy')
        m1 = np.load(args.root_path + '/ETTm1/' + df + '/' + name + '.npy')
        m2 = np.load(args.root_path + '/ETTm2/' + df + '/' + name + '.npy')
        h1 = np.load(args.root_path + '/ETTh1/' + df + '/' + name + '.npy')
        h2 = np.load(args.root_path + '/ETTh2/' + df + '/' + name + '.npy')

        print(w.shape)
        print(e.shape)
        print(t.shape)
        print(m1.shape)
        print(m2.shape)
        print(h1.shape)
        print(h2.shape)

        w = np.swapaxes(w, 1, 2)  # (B, S, T)
        w = w.reshape(-1, w.shape[-1])  # (B * S, T)

        e = np.swapaxes(e, 1, 2)  # (B, S, T)
        e = e.reshape(-1, e.shape[-1])  # (B * S, T)

        t = np.swapaxes(t, 1, 2)  # (B, S, T)
        t = t.reshape(-1, t.shape[-1])  # (B * S, T)

        m1 = np.swapaxes(m1, 1, 2)  # (B, S, T)
        m1 = m1.reshape(-1, m1.shape[-1])  # (B * S, T)

        m2 = np.swapaxes(m2, 1, 2)  # (B, S, T)
        m2 = m2.reshape(-1, m2.shape[-1])  # (B * S, T)

        h1 = np.swapaxes(h1, 1, 2)  # (B, S, T)
        h1 = h1.reshape(-1, h1.shape[-1])  # (B * S, T)

        h2 = np.swapaxes(h2, 1, 2)  # (B, S, T)
        h2 = h2.reshape(-1, h2.shape[-1])  # (B * S, T)


        data = np.concatenate((w, e, t, m1, m2, h1, h2), axis=0)
        data = np.expand_dims(data, -1)

        print(data.shape)
        print('------------------')

        np.save(args.save_path + df + '/' + name + '.npy', data)


def main(args):
    data_folders = ['Tin96_Tout96', 'Tin96_Tout192', 'Tin96_Tout336', 'Tin96_Tout720']
    check_and_save_codebook(args, data_folders)

    concatenate(args, data_folders, 'test_x_codes')
    concatenate(args, data_folders, 'test_x_original')
    concatenate(args, data_folders, 'test_y_codes')
    concatenate(args, data_folders, 'test_y_original')

    concatenate(args, data_folders, 'val_x_codes')
    concatenate(args, data_folders, 'val_x_original')
    concatenate(args, data_folders, 'val_y_codes')
    concatenate(args, data_folders, 'val_y_original')

    concatenate(args, data_folders, 'train_x_codes')
    concatenate(args, data_folders, 'train_x_original')
    concatenate(args, data_folders, 'train_y_codes')
    concatenate(args, data_folders, 'train_y_original')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--save_path', type=str, default='./data/ETT/', help='where to save the data')
    args = parser.parse_args()
    main(args)
