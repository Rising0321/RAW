import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from minRAW.utils.utils import init_dataset, init_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:8123',
                            world_size=world_size,
                            rank=rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def train(args, model, rank, world_size, train_loader, optimizer, scheduler, epoch, critetion, sampler=None):
    if rank == 0:
        print("training_%d" % epoch)
    model.train()
    ddp_loss = torch.zeros(3).to(rank)

    if sampler:
        sampler.set_epoch(epoch)

    if rank == 0:
        train_iter = tqdm(train_loader)
        enum = enumerate(train_iter)
    else:
        enum = enumerate(train_loader)

    losses = 0
    losses_mse = 0
    losses_fake = 0
    cnt = 0

    for step, (uid, trajectory) in enum:
        trajectory = trajectory.to(rank).float()  # .bfloat16()
        # trajectory = trajectory.to(rank).bfloat16()

        data, target = trajectory[:, :-1], trajectory[:, 1:]

        optimizer.zero_grad()

        output = model(data)

        mae = F.l1_loss(output, target)
        mse = critetion(output, target)
        mse.backward()

        losses_fake += F.l1_loss(data, target)

        # if rank ==0 :
        #     print(output)

        optimizer.step()
        scheduler.step()

        if rank == 0:
            train_iter.set_postfix(loss=float(mae))

        losses += float(mae)
        losses_mse += float(mse)
        cnt += 1

    if rank == 0:
        print(
            f"Train Epoch: \t{epoch}, MAE: \t{(losses / cnt):.10f}, MSE: \t{(losses_mse / cnt):.10f}, PreMAE: \t{(losses_fake / cnt):.10f}"
        )

    return losses / cnt, losses_mse / cnt


def test(model, rank, world_size, test_loader):
    if rank == 0:
        print("testing_now")
    model.eval()

    if rank == 0:
        test_iter = tqdm(test_loader)
        enum = enumerate(test_iter)
    else:
        enum = enumerate(test_loader)

    losses = 0
    cnt = 0

    with torch.no_grad():
        for step, (uid, trajectory) in enum:
            trajectory = trajectory.to(rank).float()  # .bfloat16()
            # trajectory = trajectory.to(rank).float().bfloat16()
            data, target = trajectory[:, :-1], trajectory[:, 1:]
            # print(data)
            # print(data[0][30:40])
            output = model(data)

            # print(output.shape)

            mae = F.l1_loss(output, target)

            if rank == 0:
                test_iter.set_postfix(loss=float(mae))
            losses += mae
            cnt += 1

    if rank == 0:
        print(
            f"MAE: \t{(losses / cnt):.10f}"
        )

    return losses / cnt


def fsdp_main(rank, world_size, args, train_files, eval_files):
    setup(rank, world_size)

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model, optimizer, scheduler = init_model(rank, world_size, args)

    critetion = nn.MSELoss()

    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    file_save_name = args.save_model_path

    init_start_event.record()

    train_losse_mae = []
    train_losse_mse = []

    for epoch in range(1, args.epochs + 1):

        train_loader, eval_loader, sampler1 = init_dataset(rank, world_size, args, train_files, eval_files, epoch - 1)

        mae, mse = train(args, model, rank, world_size, train_loader, optimizer, scheduler, epoch, critetion,
                         sampler=sampler1)

        train_losse_mae.append(mae)
        train_losse_mse.append(mse)

        if epoch % 100 == 0 and args.save_model:

            # save
            if rank == 0:
                print(f"--> entering save model state")

            cpu_state = model.state_dict()

            # print(f"saving process: rank {rank}  done w state_dict")

            if rank == 0:
                print(f"--> attempting to save model prefix {epoch}")
                save_name = file_save_name + "-" + str(epoch) + ".pth"
                print(f"--> saving as model name {save_name}")
                ckpt = {'model': cpu_state,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_loss': [train_losse_mae, train_losse_mse]}
                torch.save(ckpt, save_name)

    init_end_event.record()

    cleanup()


def calc_files(walk_path):
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            if file.find("data") != -1:
                res.append(root + "/" + file)
    return res


def train_gpt2_model(args: argparse.Namespace):
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)  # cpu
    torch.cuda.manual_seed_all(1)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速

    mp.spawn(fsdp_main,
             args=(args.gpus, args, calc_files(args.train_path), calc_files(args.eval_path)),
             nprocs=args.gpus,
             join=True)


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('train', help='train GPT-2 model')

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
    group.add_argument('--epochs', default=1000, type=int,
                       help='epochs')
    group.add_argument('--layers', default=12, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=16, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=1024, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=4, type=int,
                       help='increase rate of dimensionality in bottleneck')
    group.add_argument('--dropout', default=0.0, type=float,
                       help='probability that each element is dropped')
    group = parser.add_argument_group('Training and evaluation')
    group.add_argument('--batch_train', default=64, type=int,
                       help='number of training batch size')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--lr', default=1e-4, type=float,
                       help='default learning rate')
    group.add_argument('--gamma', default=1e-10, type=float,
                       help='weight gamma')

    group = parser.add_argument_group('Saving and restoring')
    group.add_argument('--save_model_path', default='model.pth',
                       help='save trained model weights to the file')
    group.add_argument('--load_path', default="",
                       help='load_model')
    group.add_argument('--from_checkpoint', default=None,
                       help='load last training state from checkpoint file')

    group = parser.add_argument_group('Extensions')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices to use in training')
    group.add_argument('--run_validation', default=False, type=bool,
                       help='run_validation')
    group.add_argument('--save_model', default=True, type=bool,
                       help='save_model')

    parser.set_defaults(func=train_gpt2_model)
