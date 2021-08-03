import time
import argparse
import datetime
import sys
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
import threading
from tqdm import tqdm

from dataloader import *




def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

# parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
# parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
# parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
#                                                                     'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
#                                                                default='densenet161_bts')
# # Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
# parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# # Log and save
# parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
# parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
# parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
# parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# # Training
# parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
# parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
# parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
# parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
# parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
# parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
# parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
# parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
# parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
# parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# # Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# # Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
# parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
# parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
# parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
# parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='gloo') # default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
# # parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
#                                                                     'N processes per node, which has N GPUs. This is the '
#                                                                     'fastest way to use PyTorch for either single node or '
#                                                                     'multi node data parallel training', action='store_true',)
# # Online eval
# parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
# parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
# parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
# parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
# parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
# parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
# parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
#                                                                     'if empty outputs to checkpoint folder', default='')




if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


args.distributed = False
dataloader = BtsDataLoader(args, 'train')
dataloader_eval = BtsDataLoader(args, 'online_eval')


global_step = 0
steps_per_epoch = len(dataloader.data)
num_total_steps = args.num_epochs * steps_per_epoch
epoch = global_step // steps_per_epoch
print("global step:", global_step)
print("steps_per_epoch", steps_per_epoch)
print("num_total_steps:", num_total_steps)
print("epoch:", epoch)


# check loaded data(kitti)
import matplotlib.pyplot as plt
import numpy as np
def custom_imshow(img): 
    img = img.cpu().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0))) 
    plt.show()
def process(): 
    for batch_idx, sample_batched in enumerate(dataloader.data): 
    
        image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
        focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
        depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))


        """
        1. focal 이 어디에 들어가는지
        
        2. 네트워크에 image 넣고 output으로 depth 나오게 하고
        
        3. depth_gt랑 비교 => loss function
        """


        print("batch_idx:", batch_idx)
        print("shape:", image.shape)  # (batch, 3, 352, 704)
        # custom_imshow(image[0]) 




if __name__ == "__main__":
    process()