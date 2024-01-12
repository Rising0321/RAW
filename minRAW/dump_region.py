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
import pandas as pd

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

    model = init_model(rank, world_size, args, test=True)
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
                data, target = trajectory[:, 1:], trajectory[:, :-1]
                output = model(data, export_embedding=1)
                uid_list.extend(uid)
                emb_list.extend(output.float().detach().cpu().numpy())
        
    
    import datetime
    now = datetime.datetime.now()
    now_time = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"-"+str(now.hour)
    np.save("./transfer/embedding/"+now_time+"_uid_%d"%rank, uid_list)
    np.save("./transfer/embedding/"+now_time+"_embedding_%d"%rank, emb_list)

    init_end_event.record()

    cleanup()

def calc_files(walk_path):
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            if file.find("data") != -1:
                res.append(root+"/"+file)
    return res

def get_human_embedding(path):
    
    embedding_dict = {}
    
    
    for i in range(6):
        uid_list = np.load(path+"_uid_%d.npy"%i)
        emb_list = np.load(path+"_embedding_%d.npy"%i)
        
        for j in range(len(uid_list)):
            uid = int(uid_list[j])
            emb = emb_list[j]
            embedding_dict[uid] = emb
            
    return embedding_dict

def get_gps_embedding(files, args, human_embedding):

    region_dic = {}
    
    for epoch in range(1, len(files) + 1):
        
        print(epoch, len(files))
        
        train_loader, eval_loader, sampler1 = init_dataset(0, 1, args, files, files, epoch - 1)

        with torch.no_grad():
            
            for step, (uid,trajectory) in enumerate(tqdm(train_loader)):

                gps_dic = {}

                trajectory = trajectory[0]
                uid = int(uid[0])

                for gps_temp in trajectory:
                    gps = (float(gps_temp[0]),float(gps_temp[1]))

                    if gps not in gps_dic.keys():
                        gps_dic[gps] = 0
                    gps_dic[gps] += 1


                for gps in gps_dic.keys():
                    if gps_dic[gps] > 0:
                        if gps not in region_dic.keys():
                            region_dic[gps] = human_embedding[uid]
                        else:
                            region_dic[gps] += human_embedding[uid]

                # print(region_dic)
                # exit(0)

    
    gps_embedding = []
    
    for gps_ in region_dic.keys():
        gps = [gps_[0],gps_[1]]
        gps[0] = gps[0] * ((117.508217 - 115.416827) / 2)
        gps[0] = gps[0] + ((117.508217 + 115.416827) / 2)
        gps[1] = gps[1] * ((41.058964 - 39.442078) / 2)
        gps[1] = gps[1] + ((41.058964 + 39.442078) / 2)
        
        gps_embedding.append([gps[0], gps[1], region_dic[gps_]])
    
    gps_df = pd.DataFrame(gps_embedding, columns=["longitude", "latitude", "embedding"])
    gps_df["longitude"] = gps_df["longitude"].astype('float').round(6)
    gps_df["latitude"] = gps_df["latitude"].astype('float').round(6)
    
    print(gps_df.head())
    
    return gps_df

def dump_region_main(args: argparse.Namespace):
    
    
    human_embedding = get_human_embedding(args.embedding_path)
    
    files = calc_files(args.train_path)
    
    gps_df = get_gps_embedding(files, args, human_embedding)
    
    
    cid_and_gps = pd.read_csv("/workdir/script/RAW/transfer/data/cid_location_20230301.csv")
    cid_and_gps["longitude"] = cid_and_gps["longitude"].round(6)
    cid_and_gps["latitude"] = cid_and_gps["latitude"].round(6)

    sum_df = pd.merge(gps_df, cid_and_gps, on=["longitude", "latitude"])

    print(sum_df.head())
    print(sum_df.tail())

    sum_df_grouped = sum_df.groupby(["longitude", "latitude"], group_keys=False)


    sum_df = sum_df_grouped.apply(lambda x: x.iloc[0])

    sum_df = sum_df[["CID", "embedding"]]
    
    import datetime
    now = datetime.datetime.now()
    now_time = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"-"+str(now.hour)
    np.save("./transfer/embedding/"+now_time+"_cid", sum_df, allow_pickle=True)

        
def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('dump', help='train GPT-2 model')

    group = parser.add_argument_group('Corpus and vocabulary')
    group.add_argument('--train_path', required=True,
                       help='training corpus file path')
    group.add_argument('--eval_path', required=True,
                       help='evaluation corpus file path')
    group.add_argument('--embedding_path', required=True,
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

    parser.set_defaults(func=dump_region_main)
