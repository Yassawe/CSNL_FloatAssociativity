import numpy as np
import random
import torch
import os
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    seed = rank
    random.seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    total = 100
    rand_sizes = [random.randint(1, 10000) for i in range(total)]
    rand_tensors = [torch.rand(rand_sizes[i]).cuda() for i in range(total)]
    rand_tensors_fuse_sync = [a.clone() for a in rand_tensors]

    # pure sync
    for a in rand_tensors:
        dist.all_reduce(a)

    # fuse sync
    cur = 0
    while cur < total:
        fus = torch.cat((rand_tensors_fuse_sync[cur], rand_tensors_fuse_sync[cur+1]))
        dist.all_reduce(fus)
        rand_tensors_fuse_sync[cur], rand_tensors_fuse_sync[cur+1] = torch.split(
            fus, (rand_sizes[cur], rand_sizes[cur+1]))
        cur += 2

    if rank == 0:
        for i in range(total):
            x1 = rand_tensors[i].mean()
            x2 = rand_tensors_fuse_sync[i].mean()
            if x1 != x2:
                d12 = torch.max(torch.abs(rand_tensors[i] - rand_tensors_fuse_sync[i]))
                print('found difference {}: {} vs {}; diffmax {}'.format(i, x1, x2, d12))

    all_tensors = torch.cat(rand_tensors)
    all_tensors_fuse_sync = torch.cat(rand_tensors_fuse_sync)

    print('[rank {}] mean {} vs {}'.format(
        rank, 
        torch.mean(all_tensors).item(), 
        torch.mean(all_tensors_fuse_sync).item()))

def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_LL_THRESHOLD'] = '0'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 4
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()