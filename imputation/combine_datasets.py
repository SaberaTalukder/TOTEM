import numpy as np
import os


def main():
    base_path = 'imputation/data/'

    all_train_notrevin_x = []
    all_train_revin_x = []

    all_val_notrevin_x = []
    all_val_revin_x = []

    all_test_notrevin_x = []
    all_test_revin_x = []

    for datatype in ['electricity/', 'ETTh1/', 'ETTh2/', 'ETTm1/', 'ETTm2/', 'weather/']:

        all_train_notrevin_x.append(np.load(base_path + datatype + 'train_notrevin_x.npy'))
        all_train_revin_x.append(np.load(base_path + datatype + 'train_revin_x.npy'))

        all_val_notrevin_x.append(np.load(base_path + datatype + 'val_notrevin_x.npy'))
        all_val_revin_x.append(np.load(base_path + datatype + 'val_revin_x.npy'))

        all_test_notrevin_x.append(np.load(base_path + datatype + 'test_notrevin_x.npy'))
        all_test_revin_x.append(np.load(base_path + datatype + 'test_revin_x.npy'))

    print('starting')

    all_train_notrevin_x_arr = np.concatenate(all_train_notrevin_x, axis=0)
    all_train_revin_x_arr = np.concatenate(all_train_revin_x, axis=0)

    print('train_done')

    all_val_notrevin_x_arr = np.concatenate(all_val_notrevin_x, axis=0)
    all_val_revin_x_arr = np.concatenate(all_val_revin_x, axis=0)

    print('val_done')

    all_test_notrevin_x_arr = np.concatenate(all_test_notrevin_x, axis=0)
    all_test_revin_x_arr = np.concatenate(all_test_revin_x, axis=0)

    print('test_done')

    if not os.path.exists(base_path + 'all'):
        os.makedirs(base_path + 'all')

    np.save(base_path + 'all/train_notrevin_x.npy', all_train_notrevin_x_arr)
    np.save(base_path + 'all/train_revin_x.npy', all_train_revin_x_arr)

    np.save(base_path + 'all/val_notrevin_x.npy', all_val_notrevin_x_arr)
    np.save(base_path + 'all/val_revin_x.npy', all_val_revin_x_arr)

    np.save(base_path + 'all/test_notrevin_x.npy', all_test_notrevin_x_arr)
    np.save(base_path + 'all/test_revin_x.npy', all_test_revin_x_arr)


if __name__ == '__main__':
    main()