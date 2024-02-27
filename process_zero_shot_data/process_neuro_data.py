import argparse
import numpy as np
import pdb
import xarray as xr
import os


def pick_patient(pt, args):
    base_path = args.base_path

    if pt == 2:
        path = base_path + 'EC02_ecog_data.nc'
        # exclude bad electrodes and examples (aka instances)
        elec_exclude = [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]
        train_instance_exclude = [10, 83, 84, 86, 168, 209, 259, 260, 261, 262, 267, 268, 269, 270, 274, 289]
        test_instance_exclude = [38, 39, 82]
        num_elecs = 72

    elif pt == 5:
        path = base_path + 'EC05_ecog_data.nc'
        elec_exclude = []
        # exclude bad electrodes and examples (aka instances)
        train_instance_exclude = [25, 26, 42, 87, 91, 92, 93]
        test_instance_exclude = [29, 41, 45, 81, 97, 98, 101, 120, 121, 122, 123, 124, 125, 126,
                                 133, 137, 148, 151, 195, 208, 210, 212, 213, 214, 217, 218, 219, 224, 225, 227, 231,
                                 233, 234, 235, 252, 255]
        num_elecs = 106

    pt = xr.open_dataset(path).to_array()[0, :, :, :]
    last_day = np.unique(pt.events)[-1]
    bool_mask_not_last_day = pt.events != last_day
    bool_mask_last_day = pt.events == last_day

    return pt, bool_mask_not_last_day, bool_mask_last_day, elec_exclude, train_instance_exclude, test_instance_exclude


def return_data_and_labels(xarr, mask, elec_exclude, instances_exclude):
    days = np.array(xarr[mask][:, :-1, :])
    # Do 0 at the end becuase all time steps are given the same label
    days_labels = np.array(xarr[mask][:, -1:, 0]) - 1  # do -1 to make the labels go from 0 to 1

    # get rid of bad electrodes
    good_elec_data = np.delete(days, elec_exclude, axis=1)

    # get rid of bad instances
    good_inst_good_elec_data = np.delete(good_elec_data, instances_exclude, axis=0)
    good_inst_good_elec_data_labels = np.delete(days_labels, instances_exclude, axis=0)

    print(good_inst_good_elec_data.shape)
    print(good_inst_good_elec_data_labels.shape)
    print(np.unique(good_inst_good_elec_data_labels))

    return good_inst_good_elec_data, good_inst_good_elec_data_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--patient_num', type=int, default=1, help='random seed')
    parser.add_argument('--base_path', type=str, default='', help='base path to nc files')
    parser.add_argument('--save_path', type=str, default='', help='place to save processed files')

    args = parser.parse_args()

    # data resampled to 250 Hz, and then they generate 2s segments centered around each event
    # Rest = 1 and Move = 2 --> intially but we -1 so --> Rest = 0 and Move = 1
    data_arr, bool_mask_not_last_day, bool_mask_last_day, elec_exclude, train_instance_exclude, test_instance_exclude = pick_patient(args.patient_num, args)
    clean_train_val_data, clean_train_val_labels = return_data_and_labels(data_arr, bool_mask_not_last_day, elec_exclude, train_instance_exclude)
    clean_test_data, clean_test_labels = return_data_and_labels(data_arr, bool_mask_last_day, elec_exclude, test_instance_exclude)

    # 90% becomes train...
    cutoff = int(clean_train_val_data.shape[0] * 0.9)

    clean_train_data = clean_train_val_data[0:cutoff]
    clean_train_labels = clean_train_val_labels[0:cutoff]

    # ... and 10% becomes val
    clean_val_data = clean_train_val_data[cutoff:]
    clean_val_labels = clean_train_val_labels[cutoff:]

    clean_train_data = np.swapaxes(clean_train_data, 1, 2)  # make it example, time, sensor
    clean_val_data = np.swapaxes(clean_val_data, 1, 2)  # make it example, time, sensor
    clean_test_data = np.swapaxes(clean_test_data, 1, 2)  # make it example, time, sensor

    print('-----------------------')
    print(clean_train_data.shape, clean_train_labels.shape)
    print(clean_val_data.shape, clean_val_labels.shape)
    print(clean_test_data.shape, clean_test_labels.shape)

    save_path = args.save_path + 'pt' + str(args.patient_num) + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(clean_train_data.shape)
    print((clean_train_data.shape[0]*clean_train_data.shape[2]))
    print('---------')

    print(clean_val_data.shape)
    print((clean_val_data.shape[0] * clean_val_data.shape[2]))
    print('---------')

    print(clean_test_data.shape)
    print((clean_test_data.shape[0] * clean_test_data.shape[2]))
    print('---------')

    print(clean_train_data.shape[0] + clean_val_data.shape[0] + clean_test_data.shape[0])
    print((clean_train_data.shape[0]*clean_train_data.shape[2]) \
          + (clean_val_data.shape[0]*clean_val_data.shape[2]) \
          + (clean_test_data.shape[0]*clean_test_data.shape[2]))

    np.save(save_path + 'train_data.npy', clean_train_data)
    np.save(save_path + 'train_labels.npy', clean_train_labels)

    np.save(save_path + 'val_data.npy', clean_val_data)
    np.save(save_path + 'val_labels.npy', clean_val_labels)

    np.save(save_path + 'test_data.npy', clean_test_data)
    np.save(save_path + 'test_labels.npy', clean_test_labels)
