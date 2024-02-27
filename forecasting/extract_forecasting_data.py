import argparse
import numpy as np
import os
import pdb
import torch

from data_provider.data_factory import data_provider
from lib.models.revin import RevIN


class ExtractData:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:' + str(self.args.gpu)
        self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
        self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def one_loop_forecasting(self, loader, vqvae_model):
        x_original_all = []
        y_original_all = []
        x_code_ids_all = []
        y_code_ids_all = []
        x_reverted_all = []
        y_reverted_all = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if i == 0:
                if batch_x.shape[-1] == batch_y.shape[-1]:
                    num_sensors = batch_x.shape[-1]
                else:
                    pdb.set_trace()

            x_original_all.append(batch_x)
            y_original_all.append(batch_y)

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            # data going into revin should have dim:[bs x time x nvars]
            x_in_revin_space = self.revin_layer_x(batch_x, "norm")
            y_in_revin_space = self.revin_layer_y(batch_y, "norm")

            # expects time to be dim [bs x nvars x time]
            x_codes, x_code_ids, codebook = time2codes(x_in_revin_space.permute(0, 2, 1), self.args.compression_factor, vqvae_model.encoder, vqvae_model.vq)
            y_codes, y_code_ids, codebook = time2codes(y_in_revin_space.permute(0, 2, 1), self.args.compression_factor, vqvae_model.encoder, vqvae_model.vq)

            x_code_ids_all.append(np.array(x_code_ids.detach().cpu()))
            y_code_ids_all.append(np.array(y_code_ids.detach().cpu()))

            # expects code to be dim [bs x nvars x compressed_time]
            x_predictions_revin_space, x_predictions_original_space = codes2time(x_code_ids, codebook, self.args.compression_factor, vqvae_model.decoder, self.revin_layer_x)
            y_predictions_revin_space, y_predictions_original_space = codes2time(y_code_ids, codebook, self.args.compression_factor, vqvae_model.decoder, self.revin_layer_y)

            x_reverted_all.append(np.array(x_predictions_original_space.detach().cpu()))
            y_reverted_all.append(np.array(y_predictions_original_space.detach().cpu()))

        x_original_arr = np.concatenate(x_original_all, axis=0)
        y_original_arr = np.concatenate(y_original_all, axis=0)

        x_code_ids_all_arr = np.concatenate(x_code_ids_all, axis=0)
        y_code_ids_all_arr = np.concatenate(y_code_ids_all, axis=0)

        x_reverted_all_arr = np.concatenate(x_reverted_all, axis=0)
        y_reverted_all_arr = np.concatenate(y_reverted_all, axis=0)

        data_dict = {}
        data_dict['x_original_arr'] = x_original_arr
        data_dict['y_original_arr'] = y_original_arr

        data_dict['x_code_ids_all_arr'] = np.swapaxes(x_code_ids_all_arr, 1, 2) # order will be [bs x compressed_time x sensors)
        data_dict['y_code_ids_all_arr'] = np.swapaxes(y_code_ids_all_arr, 1, 2) # order will be [bs x compressed_time x sensors)

        data_dict['x_reverted_all_arr'] = x_reverted_all_arr
        data_dict['y_reverted_all_arr'] = y_reverted_all_arr
        data_dict['codebook'] = np.array(codebook.detach().cpu())

        # Check to make sure sensors are last
        if data_dict['x_original_arr'].shape[-1] == num_sensors:
            if data_dict['y_original_arr'].shape[-1] == num_sensors:
                if data_dict['x_code_ids_all_arr'].shape[-1] == num_sensors:
                    if data_dict['x_reverted_all_arr'].shape[-1] == num_sensors:
                        if data_dict['y_reverted_all_arr'].shape[-1] == num_sensors:
                            print('Sensors are last')
                        else:
                            pdb.set_trace()
                    else:
                        pdb.set_trace()
                else:
                    pdb.set_trace()
            else:
                pdb.set_trace()
        else:
            pdb.set_trace()

        print(data_dict['x_original_arr'].shape, data_dict['y_original_arr'].shape)
        print(data_dict['x_code_ids_all_arr'].shape, data_dict['y_code_ids_all_arr'].shape)
        print(data_dict['x_reverted_all_arr'].shape, data_dict['y_reverted_all_arr'].shape)
        print(data_dict['codebook'].shape)

        return data_dict


    def extract_data(self):
        device = 'cuda:' + str(args.gpu)
        vqvae_model = torch.load(args.trained_vqvae_model_path)
        vqvae_model.to(device)
        vqvae_model.eval()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if args.classifiy_or_forecast == 'forecast':
            print('FORECASTING')

            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)

            # These have dimension [bs, ntime, nvars]
            print('-------------TRAIN-------------')
            train_data_dict = self.one_loop_forecasting(train_loader, vqvae_model)
            save_files_forecasting(self.args.save_path, train_data_dict, 'train', save_codebook=True)

            print('-------------Val-------------')
            val_data_dict = self.one_loop_forecasting(vali_loader, vqvae_model)
            save_files_forecasting(self.args.save_path, val_data_dict, 'val', save_codebook=False)

            print('-------------Test-------------')
            test_data_dict = self.one_loop_forecasting(test_loader, vqvae_model)
            save_files_forecasting(self.args.save_path, test_data_dict, 'test', save_codebook=True)


def save_files_forecasting(path, data_dict, mode, save_codebook):
    np.save(path + mode + '_x_original.npy', data_dict['x_original_arr'])
    np.save(path + mode + '_y_original.npy', data_dict['y_original_arr'])
    np.save(path + mode + '_x_codes.npy', data_dict['x_code_ids_all_arr'])
    np.save(path + mode + '_y_codes.npy', data_dict['y_code_ids_all_arr'])

    if save_codebook:
        np.save(path + 'codebook.npy', data_dict['codebook'])


def time2codes(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
    '''
    Args:
        revin_data: [bs x nvars x pred_len or seq_len]
        compression_factor: int
        vqvae_model: trained vqvae model
        use_grad: bool, if True use gradient, if False don't use gradients

    Returns:
        codes: [bs, nvars, code_dim, compressed_time]
        code_ids: [bs, nvars, compressed_time]
        embedding_weight: [num_code_words, code_dim]

    Helpful VQVAE Comments:
        # Into the vqvae encoder: batch.shape: [bs x seq_len] i.e. torch.Size([256, 12])
        # into the quantizer: z.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
        # into the vqvae decoder: quantized.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
        # out of the vqvae decoder: data_recon.shape: [bs x seq_len] i.e. torch.Size([256, 12])
    '''

    bs = revin_data.shape[0]
    nvar = revin_data.shape[1]
    T = revin_data.shape[2]  # this can be either the prediction length or the sequence length
    compressed_time = int(T / compression_factor)  # this can be the compressed time of either the prediction length or the sequence length

    with torch.no_grad():
        flat_revin = revin_data.reshape(-1, T)  # flat_y: [bs * nvars, T]
        latent = vqvae_encoder(flat_revin.to(torch.float), compression_factor)  # latent_y: [bs * nvars, code_dim, compressed_time]
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)  # quantized: [bs * nvars, code_dim, compressed_time]
        code_dim = quantized.shape[-2]
        codes = quantized.reshape(bs, nvar, code_dim, compressed_time)  # codes: [bs, nvars, code_dim, compressed_time]
        code_ids = encoding_indices.view(bs, nvar, compressed_time)  # code_ids: [bs, nvars, compressed_time]

    return codes, code_ids, embedding_weight


def codes2time(code_ids, codebook, compression_factor, vqvae_decoder, revin_layer):
    '''
    Args:
        code_ids: [bs x nvars x compressed_pred_len]
        codebook: [num_code_words, code_dim]
        compression_factor: int
        vqvae_model: trained vqvae model
        use_grad: bool, if True use gradient, if False don't use gradients
        x_or_y: if 'x' use revin_denorm_x if 'y' use revin_denorm_y
    Returns:
        predictions_revin_space: [bs x original_time_len x nvars]
        predictions_original_space: [bs x original_time_len x nvars]
    '''
    # print('CHECK in codes2time - should be TRUE:', vqvae_decoder.training)
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)

    with torch.no_grad():
        # scatter the label with the codebook
        one_hot_encodings = torch.zeros(int(bs * nvars * compressed_len), num_code_words, device=device)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device),1)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        quantized = torch.matmul(one_hot_encodings, torch.tensor(codebook)).view(input_shape)  # quantized: [bs * nvars, compressed_pred_len, code_dim]
        quantized_swaped = torch.swapaxes(quantized, 1,2)  # quantized_swaped: [bs * nvars, code_dim, compressed_pred_len]
        prediction_recon = vqvae_decoder(quantized_swaped.to(device), compression_factor)  # prediction_recon: [bs * nvars, pred_len]
        prediction_recon_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])  # prediction_recon_reshaped: [bs x nvars x pred_len]
        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1,2)  # prediction_recon_nvars_last: [bs x pred_len x nvars]
        predictions_original_space = revin_layer(predictions_revin_space, 'denorm')  # predictions:[bs x pred_len x nvars]

    return predictions_revin_space, predictions_original_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

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
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # Save Location
    parser.add_argument('--save_path', type=str, default=False,
                        help='folder ending in / where we want to save the revin data to')
    parser.add_argument('--trained_vqvae_model_path', type=str, required=True,
                        help='the data initialization location (gpu for indiv pts, cpu for all pts) defaults to cpu')
    parser.add_argument('--compression_factor', type=int, required=True, help='compression_factor')

    parser.add_argument('--classifiy_or_forecast', type=str, required=True, help='compression_factor')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = ExtractData
    exp = Exp(args)  # set experiments
    exp.extract_data()