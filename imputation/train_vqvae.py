import argparse
import comet_ml
import json
import numpy as np
import os
import pdb
import random
import time
import torch

from lib.models import get_model_class
from time import gmtime, strftime
from torch.utils.data import TensorDataset


def main(device, config, save_dir, logger, data_init_loc, args):
    # Create/overwrite checkpoints folder and results folder
    if os.path.exists(os.path.join(save_dir, 'checkpoints')):
        print('Checkpoint Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        pdb.set_trace()
    else:
        os.makedirs(os.path.join(save_dir, 'checkpoints'))


    logger.log_parameters(config)

    # Run start training
    vqvae_config, summary = start_training(device=device, vqvae_config=config['vqvae_config'], save_dir=save_dir,
                                           logger=logger, data_init_loc=data_init_loc, args=args)

    # Save config file
    config['vqvae_config'] = vqvae_config
    print('CONFIG FILE TO SAVE:', config)

    # Create Configs folder
    if os.path.exists(os.path.join(save_dir, 'configs')):
        print('Saved Config Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        pdb.set_trace()
    else:
        os.makedirs(os.path.join(save_dir, 'configs'))

    # Save the json copy
    with open(os.path.join(save_dir, 'configs', 'config_file.json'), 'w+') as f:
        json.dump(config, f, indent=4)

    # Save the Master File
    summary['log_path'] = os.path.join(save_dir)
    master['summaries'] = summary
    print('MASTER FILE:', master)
    with open(os.path.join(save_dir, 'master.json'), 'w') as f:
        json.dump(master, f, indent=4)


def start_training(device, vqvae_config, save_dir, logger, data_init_loc, args):
    # Create summary dictionary
    summary = {}
    general_seed = args.seed
    summary['general_seed'] = general_seed
    torch.manual_seed(general_seed)
    random.seed(general_seed)
    np.random.seed(general_seed)
    # if use another random library need to set that seed here too

    torch.backends.cudnn.deterministic = True

    summary['data initialization location'] = data_init_loc
    summary['device'] = device  # add the cpu/gpu to the summary

    # Setup model
    model_class = get_model_class(vqvae_config['model_name'].lower())
    model = model_class(vqvae_config)  # Initialize model

    print('Total # trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if vqvae_config['pretrained']:
        # pretrained needs to be the path to the trained model if you want it to load
        model = torch.load(vqvae_config['pretrained'])  # Get saved pytorch model.
    summary['vqvae_config'] = vqvae_config  # add the model information to the summary

    # Start training the model
    start_time = time.time()
    model = train_model(model, device, vqvae_config, save_dir, logger, args=args)

    # Save full pytorch model
    torch.save(model, os.path.join(save_dir, 'checkpoints/final_model.pth'))

    # Save and return
    summary['total_time'] = round(time.time() - start_time, 3)
    return vqvae_config, summary


def train_model(model, device, vqvae_config, save_dir, logger, args):
    # Set the optimizer
    optimizer = model.configure_optimizers(lr=vqvae_config['learning_rate'])

    # Setup model (send to device, set to train)
    model.to(device)
    start_time = time.time()

    print('BATCHSIZE:', args.batchsize)
    train_loader, vali_loader, test_loader = create_datloaders(batchsize=args.batchsize, dataset=vqvae_config["dataset"], base_path=args.base_path, revined_data=args.revined_data)

    # do + 0.5 to ciel it
    for epoch in range(int((vqvae_config['num_training_updates']/len(train_loader)) + 0.5)):
        model.train()
        # Do masking in the loop
        for i, (batch_x) in enumerate(train_loader):
            tensor_all_data_in_batch = torch.tensor(batch_x, dtype=torch.float, device=device)
            # random mask
            B, T = batch_x.shape
            mask = torch.rand((B, T)).to(device)
            mask[mask <= args.mask_ratio] = 0  # masked
            mask[mask > args.mask_ratio] = 1  # remained
            inp = tensor_all_data_in_batch.masked_fill(mask == 0, 0)

            loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, encoding_indices, encodings = \
                model.shared_eval(tensor_all_data_in_batch, inp, optimizer, 'train', comet_logger=logger)

            if epoch % 10000 == 0:
                comet_logger.log_metric('train_vqvae_loss_each_batch', loss.item())
                comet_logger.log_metric('train_vqvae_vq_loss_each_batch', vq_loss.item())
                comet_logger.log_metric('train_vqvae_recon_loss_each_batch', recon_error.item())
                comet_logger.log_metric('train_vqvae_perplexity_each_batch', perplexity.item())

        # # uncomment if you want the validation
        # if epoch % 1000000 == 0:
        #     with (torch.no_grad()):
        #         model.eval()
        #         for i, (batch_x) in enumerate(vali_loader):
        #             tensor_all_data_in_batch = torch.tensor(batch_x, dtype=torch.float, device=device)
        #
        #             # # random mask
        #             B, T = batch_x.shape
        #             mask = torch.rand((B, T)).to(device)
        #             mask[mask <= args.mask_ratio] = 0  # masked
        #             mask[mask > args.mask_ratio] = 1  # remained
        #             inp = tensor_all_data_in_batch.masked_fill(mask == 0, 0)
        #
        #             val_loss, val_vq_loss, val_recon_error, val_x_recon, val_perplexity, val_embedding_weight, \
        #                 val_encoding_indices, val_encodings = \
        #                 model.shared_eval(tensor_all_data_in_batch, inp, optimizer, 'val', comet_logger=logger)

        if epoch % 1000000 == 0:
            # save the model checkpoints locally and to comet
            torch.save(model, os.path.join(save_dir, f'checkpoints/model_epoch_{epoch}.pth'))
            print('Saved model from epoch ', epoch)

    print('total time: ', round(time.time() - start_time, 3))
    return model


def create_datloaders(batchsize=100, dataset="dummy", base_path='dummy', revined_data=False):

    if dataset == 'weather':
        print('weather')
        full_path = base_path + '/weather'

    elif dataset == 'electricity':
        print('electricity')
        full_path = base_path + '/electricity'

    elif dataset == 'traffic':
        print('traffic')
        full_path = base_path + '/traffic'

    elif dataset == 'ETTh1':
        print('ETTh1')
        full_path = base_path + '/ETTh1'

    elif dataset == 'ETTm1':
        print('ETTm1')
        full_path = base_path + '/ETTm1'

    elif dataset == 'ETTh2':
        print('ETTh2')
        full_path = base_path + '/ETTh2'

    elif dataset == 'ETTm2':
        print('ETTm2')
        full_path = base_path + '/ETTm2'

    elif dataset == 'all':
        print('all')
        full_path = base_path + '/all'

    else:
        print('Not done yet')
        pdb.set_trace()


    if revined_data == 'False':
        train_data = np.load(os.path.join(full_path, "train_notrevin_x.npy"), allow_pickle=True)
        val_data = np.load(os.path.join(full_path, "val_notrevin_x.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(full_path, "test_notrevin_x.npy"), allow_pickle=True)

    elif revined_data == 'True':
        train_data = np.load(os.path.join(full_path, "train_revin_x.npy"), allow_pickle=True)
        val_data = np.load(os.path.join(full_path, "val_revin_x.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(full_path, "test_revin_x.npy"), allow_pickle=True)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batchsize,
                                                   shuffle=True,
                                                   num_workers=10,
                                                   drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    #create argument parser to read in from the python terminal call
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        required=False, default='',
                        help='path to specific config file once already in the config folder')
    parser.add_argument('--model_init_num_gpus', type=int,
                        required=False, default=0,
                        help='number of gpus to use, 0 indexed, so if you want 1 gpu say 0')
    parser.add_argument('--data_init_cpu_or_gpu', type=str,
                        required=False,
                        help='the data initialization location')
    parser.add_argument('--comet_log', action='store_true',
                        required=False,
                        help='whether to log to comet online')
    parser.add_argument('--comet_tag', type=str,
                        required=False,
                        help='the experimental tag to add to comet - this should be the person running the exp')
    parser.add_argument('--comet_name', type=str,
                        required=False,
                        help='the experiment name to add to comet')
    parser.add_argument('--save_path', type=str,
                        required=False,
                        help='where were going to save the checkpoints')
    parser.add_argument('--base_path', type=str,
                        default=False, help='saved revin data to train model')
    parser.add_argument('--batchsize', type=int,
                        required=True,
                        help='batchsize')

    parser.add_argument('--mask_ratio', type=float,required=True, help='how much to mask the data')

    parser.add_argument('--revined_data', type=str, required=True, help='if true use revin, if false do something else')

    parser.add_argument('--seed', type=int, required=True, help='the seed to use')

    args = parser.parse_args()

    # Get config file
    config_file = args.config_path
    print('Config folder:\t {}'.format(config_file))

    # Load JSON config file
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(' Running Config:', config_file)

    # save directory --> will be identically named to config structure
    save_folder_name = ('CD' + str(config['vqvae_config']['embedding_dim']) +
                        '_CW' + str(config['vqvae_config']['num_embeddings']) +
                        '_CF' + str(config['vqvae_config']['compression_factor']) +
                        '_BS' + str(args.batchsize) +
                        '_ITR' + str(config['vqvae_config']['num_training_updates']) +
                        '_seed' + str(args.seed) +
                        '_maskratio' + str(args.mask_ratio))

    save_dir = args.save_path + save_folder_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    master = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'config file': config_file,
        'save directory': save_dir,
        'gpus': args.model_init_num_gpus,
    }

    # set up comet logger
    if args.comet_log:
        # Create an experiment with your api key
        comet_logger = comet_ml.Experiment(
            api_key=config['comet_config']['api_key'],
            project_name=config['comet_config']['project_name'],
            workspace=config['comet_config']['workspace'],
        )
        comet_logger.add_tag(args.comet_tag)
        comet_logger.set_name(args.comet_name)
    else:
        print('PROBLEM: not saving to comet')
        comet_logger = None
        pdb.set_trace()

    # Set up GPU / CPU
    if torch.cuda.is_available() and args.model_init_num_gpus >= 0:
        assert args.model_init_num_gpus < torch.cuda.device_count()  # sanity check
        device = 'cuda:{:d}'.format(args.model_init_num_gpus)
    else:
        device = 'cpu'

    # Where to init data for training (cpu or gpu) -->  will be trained wherever args.model_init_num_gpus says
    if args.data_init_cpu_or_gpu == 'gpu':
        data_init_loc = device  # we do this so that data_init_loc will have the correct cuda:X if gpu
    else:
        data_init_loc = 'cpu'

    # call main
    main(device, config, save_dir, comet_logger, data_init_loc, args)
