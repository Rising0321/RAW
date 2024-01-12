import argparse
import os
import argparse
import functools
from distutils.version import LooseVersion
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import nccl
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR
import random
import torch.distributed as dist
import torch.multiprocessing as mp

from minRAW.utils.utils import init_dataset, init_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"


def setup(rank, world_size):
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0) #cpu
    torch.cuda.manual_seed_all(0)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    # torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:8123',
                            world_size=world_size,
                            rank=rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()

def fsdp_main(rank, world_size, args, train_files, eval_files):
    setup(rank, world_size)

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = init_model(rank, world_size, args, test=True)

    init_start_event.record()
    
    prediction = []
    truth = []
    
    with torch.no_grad():
        
        model.eval()
        
        for epoch in range(1, args.epochs + 1):

            train_loader, eval_loader, sampler1 = init_dataset(rank, world_size, args, train_files, eval_files, epoch - 1)

            for (uid, trajectory) in tqdm(train_loader):

                trajectory = trajectory.to(rank).float()

                data, target = trajectory[:, :-1], trajectory[:, 1:]
                
                now_target = torch.tensor(data)
                
                for i in range(args.gen_len):
                    output = model(data)
                    data[:, args.seq_len - args.gen_len + i - 1] = output[:, args.seq_len - args.gen_len + i - 2].cuda()
                
                # output = model(data)
                
                prediction.extend(data[:,-args.gen_len:].float().detach().cpu().numpy() * 100 * 1000 )
                truth.extend(now_target[:,-args.gen_len:].detach().cpu().numpy() * 100 * 1000 )
    
    import datetime
    now_time = str(datetime.datetime.now()).replace(" ","-")
    np.save("./autoregression/data/"+now_time+"_prediction", prediction)
    np.save("./autoregression/data/"+now_time+"_truth", truth)
    
    output = np.load("./autoregression/data/"+now_time+"_prediction.npy").reshape([-1, 10, 2])

    truth = np.load("./autoregression/data/"+now_time+"_truth.npy").reshape([-1, 10, 2])

    def calc(reshaped_prediction, reshaped_label):
        mse = mean_squared_error(reshaped_prediction, reshaped_label)
        mae = mean_absolute_error(reshaped_prediction, reshaped_label)
        pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
        print(mae, mse, pcc)
    
    for i in range(0,11):
        calc(output[:,-i], truth[:, -i])
       
    init_end_event.record()

    cleanup()

def calc_files(walk_path):
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            if file.find("data") != -1:
                res.append(root+"/"+file)
    return res

def generate_sentence(args: argparse.Namespace):
    
    mp.spawn(fsdp_main,
             args=(args.gpus, args, calc_files(args.train_path), calc_files(args.eval_path)),
             nprocs=args.gpus,
             join=True)



def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('generate', help='train GPT-2 model')

    group = parser.add_argument_group('Corpus and vocabulary')
    group.add_argument('--train_path', required=True,
                       help='training corpus file path')
    group.add_argument('--eval_path', required=True,
                       help='evaluation corpus file path')

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seed', default=64, type=int,
                       help='random_seed')
    group.add_argument('--seq_len', default=64, type=int,
                       help='maximum sequence length')
    group.add_argument('--num_iter', default=1000, type=int,
                       help='generate_number')
    group.add_argument('--gen_len', default=10, type=int,
                       help='generate_length')
    group.add_argument('--layers', default=12, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=16, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=1024, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=4, type=int,
                       help='increase rate of dimensionality in bottleneck')
    group.add_argument('--dropout', default=0.1, type=float,
                       help='probability that each element is dropped')
    group.add_argument('--epochs', default=1000, type=int,
                       help='epochs')

    group = parser.add_argument_group('Training and evaluation')
    group.add_argument('--batch_train', default=64, type=int,
                       help='number of training batch size')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--lr', default=1e-4, type=float,
                       help='default learning rate')
    group.add_argument('--gamma', default=1e-2, type=float,
                       help='weight gamma')

    group = parser.add_argument_group('Saving and restoring')
    group.add_argument('--save_model_path', default='model.pth',
                       help='save trained model weights to the file')

    group.add_argument('--load_path', default="./minRAW/log/-1000.pth",
                       help='load last training state from checkpoint file')
    group.add_argument('--save_path', default="./truth_trajectories.npy",
                       help='load last training state from checkpoint file')

    group = parser.add_argument_group('Extensions')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices to use in training')
    group.add_argument('--run_validation', default=True, type=bool,
                       help='run_validation')
    group.add_argument('--save_model', default=True, type=bool,
                       help='save_model')

    parser.set_defaults(func=generate_sentence)
