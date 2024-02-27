import argparse
import numpy as np
import os
import pdb
import torch
import torch.nn as nn

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def revintime2codes(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
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
        # this is if your compression factor=4
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
        codes = quantized.reshape(bs, nvar, code_dim,
                                  compressed_time)  # codes: [bs, nvars, code_dim, compressed_time]
        code_ids = encoding_indices.view(bs, nvar, compressed_time)  # code_ids: [bs, nvars, compressed_time]

    return codes, code_ids, embedding_weight


def codes2timerevin(code_ids, codebook, compression_factor, vqvae_decoder):
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

    return predictions_revin_space


def create_datloaders(batchsize=100, dataset="dummy", base_path='dummy', labels_path='dummy'):

    if dataset == 'MSL' or dataset == 'PSM' or dataset == 'SMAP' or dataset == 'SMD' or dataset == 'SWAT':
        print(dataset)
        full_path = base_path

        # you can test on either the revined data or non revined data

        # this pulls in revined data
        train_data = np.load(os.path.join(full_path, "train.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(full_path, "test.npy"), allow_pickle=True)
        test_labels = np.load(os.path.join(labels_path, "test_labels_processed.npy"), allow_pickle=True)

        # # this pulls in non revined data
        # train_data = np.load(os.path.join(labels_path, "train_data_processed.npy"), allow_pickle=True)
        # test_data = np.load(os.path.join(labels_path, "train_data_processed.npy"), allow_pickle=True)
        # test_labels = np.load(os.path.join(labels_path, "test_labels_processed.npy"), allow_pickle=True)


    # zero shot datasets
    elif dataset == 'saugeen' or dataset == 'sunspot' or dataset == 'neuro2' or dataset == 'us_births' or dataset == 'neuro5':
        print(dataset)
        full_path = base_path

        # you can test on either the revined data or non revined data

        # # this pulls in revined data
        train_data = np.load(os.path.join(full_path, "train_revin.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(full_path, "test_revin_masked.npy"), allow_pickle=True)
        test_labels = np.load(os.path.join(labels_path, "test_labels_totem_shaped.npy"), allow_pickle=True)

        # # this pulls in not revined data
        # train_data = np.load(os.path.join(full_path, "train_notrevin.npy"), allow_pickle=True)
        # test_data = np.load(os.path.join(full_path, "test_notrevin_masked.npy"), allow_pickle=True)
        # test_labels = np.load(os.path.join(labels_path, "test_labels_totem_shaped.npy"), allow_pickle=True)

    else:
        print('Not done yet')
        pdb.set_trace()


    # need batch, sensor, time
    train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 1, 2)

    print('TRAIN SHAPE:', train_data.shape)
    print('TEST SHAPE:', test_data.shape)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batchsize,
                                                   shuffle=False,
                                                   num_workers=10,
                                                   drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

    test_labels_dataloader = torch.utils.data.DataLoader(test_labels,
                                                  batch_size=batchsize,
                                                  shuffle=False,
                                                  num_workers=10,
                                                  drop_last=False)

    return train_dataloader, test_dataloader, test_labels_dataloader


# Taken from GPT2 and TimesNet
# GPT2: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/9f3875dbba9f52ee534ac0902572502746f3b29d/Anomaly_Detection/utils/tools.py#L91
# TimesNet: https://github.com/thuml/Time-Series-Library/blob/8782a590950b4ee31a0d2d66fc30b761689c3636/utils/tools.py#L93
# See discussion from Anomaly Transformer for why it exists: https://github.com/thuml/Anomaly-Transformer/issues/14
# tl;dr: there is a distinction between time step anomaly detection and event anomaly detection, this function translates from time step anomaly detection to event anomaly detection.
def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def test(args):
    device = 'cuda:' + str(args.gpu)
    vqvae_model = torch.load(args.trained_vqvae_model_path)
    vqvae_model.to(device)
    vqvae_model.eval()

    train_dataloader, test_dataloader, test_labels_dataloader = create_datloaders(batchsize=2048, dataset=args.dataset, base_path=args.base_path, labels_path=args.labels_path)

    attens_energy = []

    anomaly_criterion = nn.MSELoss(reduce=False)

    # (1) statistic on the train set
    with torch.no_grad():
        for i, batch_x in enumerate(train_dataloader):
            batch_x = batch_x.float().to(device)

            # expects time to be dim [bs x nvars x time]
            x_codes, x_code_ids, codebook = revintime2codes(batch_x, args.compression_factor, vqvae_model.encoder, vqvae_model.vq)

            # expects code to be dim [bs x nvars x compressed_time]
            x_predictions_revin_space = codes2timerevin(x_code_ids, codebook, args.compression_factor, vqvae_model.decoder)

            # comes out as [bs x time x nvars]
            batch_x_swapped = np.swapaxes(batch_x, 1, 2)

            # criterion --> must be computed along the sensor dimension (e.g. dim=-1)
            score = torch.mean(anomaly_criterion(batch_x_swapped, x_predictions_revin_space), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

            if i == 0:

                if batch_x.shape[1] != args.num_vars or batch_x.shape[2] != args.seq_len:
                    print('batch x shape is wrong')
                    pdb.set_trace()

                if batch_x_swapped.shape[1] != args.seq_len or batch_x_swapped.shape[2] != args.num_vars:
                    print('batch_x_swapped shape is wrong')
                    pdb.set_trace()

                if x_predictions_revin_space.shape[1] != args.seq_len or x_predictions_revin_space.shape[2] != args.num_vars:
                    print('x_predictions_revin_space shape is wrong')
                    pdb.set_trace()

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)

    # get all test labels
    test_labels = []
    for i, batch_y in enumerate(test_labels_dataloader):
        test_labels.append(batch_y)

    # (2) find the threshold
    attens_energy = []
    for i, batch_x in enumerate(test_dataloader):
        batch_x = batch_x.float().to(device)

        # expects time to be dim [bs x nvars x time]
        x_codes, x_code_ids, codebook = revintime2codes(batch_x, args.compression_factor, vqvae_model.encoder,
                                                        vqvae_model.vq)

        # expects code to be dim [bs x nvars x compressed_time]
        x_predictions_revin_space = codes2timerevin(x_code_ids, codebook, args.compression_factor, vqvae_model.decoder)

        # comes out as [bs x time x nvars]
        batch_x_swapped = np.swapaxes(batch_x, 1, 2)

        # criterion --> must be computed along the sensor dimension (e.g. dim=-1)
        score = torch.mean(anomaly_criterion(batch_x_swapped, x_predictions_revin_space), dim=-1)
        score = score.detach().cpu().numpy()
        attens_energy.append(score)

        if i == 0:
            if batch_x.shape[1] != args.num_vars or batch_x.shape[2] != args.seq_len:
                print('batch x shape is wrong')
                pdb.set_trace()

            if batch_x_swapped.shape[1] != args.seq_len or batch_x_swapped.shape[2] != args.num_vars:
                print('batch_x_swapped shape is wrong')
                pdb.set_trace()

            if x_predictions_revin_space.shape[1] != args.seq_len or x_predictions_revin_space.shape[
                2] != args.num_vars:
                print('x_predictions_revin_space shape is wrong')
                pdb.set_trace()


    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - args.anomaly_ratio)
    print("Threshold :", threshold)

    # (3) evaluation on the test set
    pred = (test_energy > threshold).astype(int)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    # (4) detection adjustment
    # The following function is taken from GPT2 and TimesNet
    # GPT2: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/9f3875dbba9f52ee534ac0902572502746f3b29d/Anomaly_Detection/utils/tools.py#L91
    # TimesNet: https://github.com/thuml/Time-Series-Library/blob/8782a590950b4ee31a0d2d66fc30b761689c3636/utils/tools.py#L93
    # See discussion from Anomaly Transformer for why it exists: https://github.com/thuml/Anomaly-Transformer/issues/14
    # tl;dr: there is a distinction between time step anomaly detection and event anomaly detection, this function translates from time step anomaly detection to event anomaly detection.
    gt, pred = adjustment(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)
    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    parser.add_argument('--dataset', type=str, required=True, help='')

    parser.add_argument('--trained_vqvae_model_path', type=str, required=True, help='')
    parser.add_argument('--compression_factor', type=int, required=True, help='compression_factor')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--base_path', type=str, help='which data to perform oracle on', default=False)
    parser.add_argument('--labels_path', type=str, help='path to the test labels', default=False)

    parser.add_argument('--anomaly_ratio', type=float, help='anomaly in the dataset', default=False) # this is predefined for each dataset
    parser.add_argument('--seq_len', type=int, help='time series length', default=True)

    parser.add_argument('--num_vars', type=int, help='number of sensors', default=0)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    test(args)