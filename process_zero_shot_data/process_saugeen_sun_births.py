import os
import pdb
import argparse

import numpy as np
import pandas as pd
from distutils.util import strtobool
from datetime import datetime


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def dropna(x):
    return x[~np.isnan(x)]


def ltf_stride(data, input_size, output_size, stride):  # want output shape to be (examples, time, sensors)
    start = 0
    middle = input_size
    stop = input_size + output_size

    data_x = []
    data_y = []

    for i in range(0, data.shape[1]):
        if stop >= data.shape[1]:
            break

        data_x.append(data[:, start:middle])
        data_y.append(data[:, middle:stop])

        start += stride
        middle += stride
        stop += stride

    data_x_arr = np.concatenate(data_x, axis=0)
    data_y_arr = np.concatenate(data_y, axis=0)

    data_x_arr = np.expand_dims(data_x_arr, axis=-1)  # add the sensor dimension
    data_y_arr = np.expand_dims(data_y_arr, axis=-1)  # add the sensor dimension

    return data_x_arr, data_y_arr



def main(args):
    path = args.base_path

    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(path)

    timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
    timeseries_arr = np.array(timeseries)


    if 'saugeen' in path:
        in_size = 96
        out_size = 96
        data_x_arr, data_y_arr = ltf_stride(timeseries_arr, in_size, out_size, 1)

        print(data_x_arr.shape, data_y_arr.shape)

        save_path = args.save_path + '/saugeen/Tin'+ str(in_size)+ '_Tout' + str(out_size) + '/'
        print(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(save_path + 'all_x_original.npy', data_x_arr)
        np.save(save_path + 'all_y_original.npy', data_y_arr)

    elif 'us_births' in path:
        in_size = 96
        out_size = 96
        data_x_arr, data_y_arr = ltf_stride(timeseries_arr, in_size, out_size, 1)

        print(data_x_arr.shape, data_y_arr.shape)

        save_path = args.save_path + '/us_births/Tin' + str(in_size) + '_Tout' + str(out_size) + '/'
        print(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(save_path + 'all_x_original.npy', data_x_arr)
        np.save(save_path + 'all_y_original.npy', data_y_arr)

    elif 'sunspot' in path:
        in_size = 96
        out_size = 96
        data_x_arr, data_y_arr = ltf_stride(timeseries_arr, in_size, out_size, 1)

        print(data_x_arr.shape, data_y_arr.shape)

        save_path = args.save_path + '/sunspot/Tin' + str(in_size) + '_Tout' + str(out_size) + '/'
        print(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(save_path + 'all_x_original.npy', data_x_arr)
        np.save(save_path + 'all_y_original.npy', data_y_arr)



if __name__== '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_path', type=str, default='', help='base path to nc files')
    parser.add_argument('--save_path', type=str, default='', help='place to save processed files')

    args = parser.parse_args()
    main(args)


