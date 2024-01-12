from minRAW.data.trajectories import Trajectories
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.optim as optim
import torch.nn as nn
import functools


def init_dataset(rank, world_size, args, train_files, eval_files, epoch):
    # print(train_files,eval_files)
    train_dataset = Trajectories(train_files, args.seq_len, epoch)
    eval_dataset = Trajectories(eval_files, args.seq_len,epoch)

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
    sampler2 = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_train, 'sampler': sampler1}
    eval_kwargs = {'batch_size': args.batch_eval, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    eval_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, **eval_kwargs)

    return train_loader, eval_loader, sampler1


def init_model(rank, world_size, args, test=False):
    from minRAW.modeling.transformer import Transformer
    model = Transformer(layers=args.layers, seq_len=args.seq_len,
                    heads=args.heads, dims=args.dims, rate=args.rate,
                    dropout=args.dropout).to(rank)
    
    # print(model)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=2000
    )

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )
    
    
    if test or args.load_path!="":
        print("load_state_dict 0 ")
        state_dict = torch.load(args.load_path, map_location='cuda')
        # print(state_dict['model'].keys())
        new_stete_dict = {}
        for key in state_dict['model'].keys():
            new_stete_dict[key[7:]] = state_dict['model'][key]
        model.load_state_dict(new_stete_dict)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1 - step / 700000)

    model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank]).cuda()

    if test or args.load_path!="":
        print("load_state_dict 1 ")
        state_dict = torch.load(args.load_path, map_location='cuda')

        # model.load_state_dict(state_dict['model'])
        # optimizer.load_state_dict(state_dict['optimizer'])
        # scheduler.load_state_dict(state_dict['scheduler'])

    return model, optimizer, scheduler
