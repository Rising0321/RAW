import argparse
import os
import argparse
import functools
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import nccl
from torch.distributed._shard.checkpoint import load_state_dict, FileSystemReader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp

from minRAW.utils.utils import init_dataset, init_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:8128',
                            world_size=world_size,
                            rank=rank)


def cleanup():
    dist.destroy_process_group()


    
def fsdp_main(rank, world_size, args, train_files, eval_files):
    setup(rank, world_size)

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model, optimizer, scheduler  = init_model(rank, world_size, args, test=True)
    model.eval()
    
    uid_list = []
    emb_list = []
    
    init_start_event.record()
    for epoch in range(1, len(eval_files) + 1):
        
        if rank == 0:
            print("%d of %d"%(epoch, len(eval_files)))
        
        train_loader, eval_loader, sampler1 = init_dataset(rank, world_size, args, train_files, eval_files, epoch - 1)
        
        if rank == 0:
            test_iter = tqdm(train_loader)
            enum = enumerate(test_iter)
        else:
            enum = enumerate(train_loader)
        
        with torch.no_grad():
            for step, (uid,trajectory) in enum:
                
                trajectory = trajectory.to(rank).float()
                data, target = trajectory[:, :-1], trajectory[:, 1:]
                embedding, output = model(data, export_embedding=1)
                uid_list.extend(uid.long().detach().cpu().numpy())
                emb_list.extend(embedding.float().detach().cpu().numpy())
                
                mae = F.l1_loss(output, target)
                
                if rank == 0:
                    test_iter.set_postfix(loss=float(mae))
        
    
    import datetime
    now = datetime.datetime.now()
    now_time = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"-"+str(now.hour)
    np.save("./transfer/embedding/"+now_time+"_1_uid_%d"%rank, uid_list)
    np.save("./transfer/embedding/"+now_time+"_1_embedding_%d"%rank, emb_list)

    init_end_event.record()

    cleanup()

def calc_files(walk_path):
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            if file.find("data") != -1:
                res.append(root+"/"+file)
    return res

def export_embeddings_main(args: argparse.Namespace):
    
    mp.spawn(fsdp_main,
             args=(args.gpus, args, calc_files(args.train_path), calc_files(args.eval_path)),
             nprocs=args.gpus,
             join=True)
    

def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('export', help='train GPT-2 model')

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

    group.add_argument('--load_path', default="./minRAW/log/-2500.pth",
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

    parser.set_defaults(func=export_embeddings_main)
