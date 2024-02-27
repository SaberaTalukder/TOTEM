import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lib.models.revin import RevIN
from lib.models.metrics import pearsoncor


def create_time_series_dataloader(datapath="/data", batchsize=8):
    dataloaders = {}
    for split in ["test"]:
        # original time series
        timex_file = os.path.join(datapath, "%s_x_original.npy" % split)
        timex = np.load(timex_file)
        timex = torch.from_numpy(timex).to(dtype=torch.float32)
        timey_file = os.path.join(datapath, "%s_y_original.npy" % split)
        timey = np.load(timey_file)
        timey = torch.from_numpy(timey).to(dtype=torch.float32)

        # x codes
        codex_file = os.path.join(datapath, "%s_x_codes.npy" % (split))
        codex = np.load(codex_file)
        codex = torch.from_numpy(codex).to(dtype=torch.int64)

        codey_oracle_file = os.path.join(datapath, "%s_y_codes_oracle.npy" % split)
        if not os.path.exists(codey_oracle_file):
            codey_oracle_file = os.path.join(datapath, "%s_y_codes.npy" % split)
        codey_oracle = np.load(codey_oracle_file)
        codey_oracle = torch.from_numpy(codey_oracle).to(dtype=torch.int64)

        print("[Dataset][%s] %d of examples" % (split, timex.shape[0]))

        dataset = torch.utils.data.TensorDataset(timex, timey, codex, codey_oracle)
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True if split == "train" else False,
            num_workers=10,
            drop_last=True if split == "train" else False,
        )

    return dataloaders


def inference(
    data,
    model_decode,
    model_mustd,
    codebook,
    compression,
    device,
    onehot: bool = False,
    scheme: int = 2,
):
    """
    Returns:
        out: (B, Tout, Sout)
    """
    code_dim = codebook.shape[1]

    x, y, codeids_x, codeids_y_labels = data
    x = x.to(device)
    codeids_x = codeids_x.to(device)

    # prepare data
    B, TCin, Sin = codeids_x.shape
    B, TCout, Sout = codeids_y_labels.shape
    Tout = TCout * compression
    del codeids_y_labels

    _ = model_mustd.revin_in(x, "norm")
    _ = model_mustd.revin_out(y, "norm")

    # get codewords for input x
    code_ids = codeids_x.flatten()
    if onehot:
        xcodes = F.one_hot(code_ids, num_classes=codebook.shape[0])
    else:
        xcodes = codebook[code_ids]  # (B*Tin*Sin, D)
    xcodes = xcodes.reshape((B, TCin, Sin, xcodes.shape[-1]))  # (B, TCin, Sin, D)
    xcodes = torch.permute(xcodes, (0, 2, 1, 3))  # (B, Sin, TCin, D)
    xcodes = xcodes.reshape((B * Sin, TCin, xcodes.shape[-1]))  # (B * Sin, TCin, D)
    xcodes = torch.permute(xcodes, (1, 0, 2))  # (TCin, B * Sin, D)

    ytime = model_decode(xcodes)  # (B * Sout, Tout)

    # reshapes
    ytime = ytime.reshape((B, Sout, Tout))  # (B, Sout, Tout)
    ytime = torch.permute(ytime, (0, 2, 1))  # (B, Tout, Sout)

    # ------ MEAN & STD ------ #
    times = torch.permute(x, (0, 2, 1))  # (B, Sin, Tin * C)
    times = times.reshape((-1, times.shape[-1]))
    ymeanstd = model_mustd(times)

    # reshapes
    ymeanstd = ymeanstd.reshape((B, Sout, 2))  # (B, S, 2)
    ymeanstd = torch.permute(ymeanstd, (0, 2, 1))  # (B, 2, S)
    ymean = ymeanstd[:, 0, :].unsqueeze(1)  # (B, 1, S)
    ystd = ymeanstd[:, 1, :].unsqueeze(1)  # (B, 1, S)

    if scheme == 1:
        ymean = ymean + model_mustd.revin_in.mean
        ystd = ystd + model_mustd.revin_in.stdev
        ytime = ytime * ystd + ymean
    elif scheme == 2:
        ytime = model_mustd.revin_in(ytime, "denorm")
    else:
        raise ValueError("Unknown prediction scheme %d" % scheme)

    return ytime


def eval(args):
    device = torch.device("cuda:%d" % (args.cuda_id))
    torch.cuda.set_device(device)

    # -------- PARAMS ------- #
    params = get_params(args.data_type, args.data_path)
    batchsize = params["batchsize"]
    dataroot = params["dataroot"]
    Sin, Sout = params["Sin"], params["Sout"]
    is_affine_revin = False
    compression = args.compression

    datapath = dataroot

    # -------- CODEBOOK ------- #
    codebook = np.load(os.path.join(datapath, "codebook.npy"), allow_pickle=True)
    codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
    vocab_size, vocab_dim = codebook.shape

    assert vocab_size == args.codebook_size
    dim = vocab_size if args.onehot else vocab_dim

    # ------ DATA LOADERS ------- #
    dataloaders = create_time_series_dataloader(datapath=datapath, batchsize=batchsize)
    test_dataloader = dataloaders["test"]
    norcal_dataloader = None

    # ------- MODEL: XCODES TO YTIME -------- #
    model_decode = torch.load(args.model_load_path + 'decode_checkpoint.pth')

    # ------- MODEL: MuStd ----------#
    model_mustd = torch.load(args.model_load_path + 'mustd_checkpoint.pth')
    model_mustd.revin_in = RevIN(
        num_features=Sin, affine=is_affine_revin
    )  # expects as input (B, T, S)
    model_mustd.revin_out = RevIN(
        num_features=Sout, affine=is_affine_revin
    )  # expects as input (B, T, S)

    model_decode.to(device)
    model_mustd.to(device)

    if norcal_dataloader is not None:
        model_decode.eval()
        model_mustd.eval()
        running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
        total_num, total_num_c = 0.0, 0.0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():

            all_prediction_time = []

            for i, tdata in enumerate(norcal_dataloader):
                pred_time = inference(
                    tdata,
                    model_decode,
                    model_mustd,
                    codebook,
                    args.compression,
                    device,
                    onehot=args.onehot,
                    scheme=args.scheme,
                )
                labels_code = tdata[-1]
                labels_code = labels_code.to(device)
                labels_time = tdata[1]
                labels_time = labels_time.to(device)

                running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                running_mae += (pred_time - labels_time).abs().sum()
                running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                total_num += labels_time.numel()
                total_num_c += labels_time.shape[0] * labels_time.shape[2]  # B * S

                all_prediction_time.append(np.array(pred_time.detach().cpu()))

            all_prediction_time_array = np.concatenate(all_prediction_time, axis=0)

        running_mae = running_mae / total_num
        running_mse = running_mse / total_num
        running_cor = running_cor / total_num_c
        print('NORCAL NUMBERS:')
        print(f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}")


    if test_dataloader is not None:
        model_decode.eval()
        model_mustd.eval()
        running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
        total_num, total_num_c = 0.0, 0.0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():

            all_prediction_time = []

            for i, tdata in enumerate(test_dataloader):
                pred_time = inference(
                    tdata,
                    model_decode,
                    model_mustd,
                    codebook,
                    args.compression,
                    device,
                    onehot=args.onehot,
                    scheme=args.scheme,
                )
                labels_code = tdata[-1]
                labels_code = labels_code.to(device)
                labels_time = tdata[1]
                labels_time = labels_time.to(device)

                running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                running_mae += (pred_time - labels_time).abs().sum()
                running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                total_num += labels_time.numel()
                total_num_c += labels_time.shape[0] * labels_time.shape[2]  # B * S

                all_prediction_time.append(np.array(pred_time.detach().cpu()))

            all_prediction_time_array = np.concatenate(all_prediction_time, axis=0)

        running_mae = running_mae / total_num
        running_mse = running_mse / total_num
        running_cor = running_cor / total_num_c
        print('TEST NUMBERS')
        print(f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}")


def get_params(data_type, data_path):
    if data_type == "weather":
        batchsize = 128
        Sin = Sout = 21
        dataroot = data_path

    elif data_type == "traffic":
        batchsize = 16
        Sin = Sout = 862
        dataroot = data_path

    elif data_type == "electricity":
        batchsize = 32
        Sin = Sout = 321
        dataroot = data_path

    elif data_type == "ETTh1":
        batchsize = 128
        Sin = Sout = 7
        dataroot = data_path

    elif data_type == "ETTh2":
        batchsize = 128
        Sin = Sout = 7
        dataroot = data_path

    elif data_type == "ETTm1":
        batchsize = 128
        Sin = Sout = 7
        dataroot = data_path

    elif data_type == "ETTm2":
        batchsize = 128
        Sin = Sout = 7
        dataroot = data_path

    elif data_type == "all":
        batchsize = 4096
        Sin = Sout = 1
        dataroot = data_path

    elif data_type == "neuro2":
        batchsize = 512
        Sin = Sout = 72
        dataroot = data_path

    elif data_type == "neuro5":
        batchsize = 512
        Sin = Sout = 106
        dataroot = data_path

    elif data_type == "saugeen":
        batchsize = 16384
        Sin = Sout = 1
        dataroot = data_path

    elif data_type == "us_births":
        batchsize = 16384
        Sin = Sout = 1
        dataroot = data_path

    elif data_type == "sunspot":
        batchsize = 16384
        Sin = Sout = 1
        dataroot = data_path

    else:
        raise ValueError("Unknown data type %s" % (args.data_type))
    return {"dataroot": dataroot, "batchsize": batchsize, "Sin": Sin, "Sout": Sout}


def default_argument_parser():
    """
    Create a parser.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Code Prediction")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--cuda-id", default=0, type=int)
    # ---------- SEED ---------- #
    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    parser.add_argument("--seed", default=-1, type=int)
    # ---------- DATA ---------- #
    parser.add_argument("--data-type", default="weather", type=str)
    parser.add_argument("--codebook_size", default=256, type=int)
    parser.add_argument("--compression", default=4, type=int)
    parser.add_argument("--Tin", default=96, type=int)
    parser.add_argument("--Tout", default=96, type=int)
    parser.add_argument("--data_path", default='', type=str, help="path to data")

    # ----------- CHECKPOINT ------------ #
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument(
        "--checkpoint_path", default="/data/", type=str
    )
    parser.add_argument("--patience", default=3, type=int)
    # ---------- TRAINING ---------- #
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--scheduler", default="onecycle", type=str)
    parser.add_argument("--baselr", default=0.0001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--steps", default=4, type=int, help="decay LR in epochs")

    parser.add_argument(
        "--beta", default=0.1, type=float, help="beta for smoothl1 loss"
    )
    # ----------- INPUT ------------ #
    parser.add_argument(
        "--onehot",
        action="store_true",
        help="use onehot representation if true, otherwise use codes",
    )
    # ----------- SCHEME ------------ #
    parser.add_argument(
        "--scheme",
        default=1,
        type=int,
        help="1 predicts mu/std, 2 urevins the prediction (as in PatchTST)",
    )
    # ----------- ENCODER PARAMS ------------ #
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--d_hid", default=256, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--nlayers", default=4, type=int)

    parser.add_argument("--model_load_path", default='', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    eval(args)