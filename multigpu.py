import os
import glob
from collections import deque
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import contextmanager

@contextmanager
def main_process_only():
    if is_main_process():
        yield
    else:
        yield

@contextmanager
def main_process_first():
    if not dist.is_initialized() or dist.get_rank() == 0:
        yield
    else:
        dist.barrier()
        yield
    if dist.is_initialized():
        dist.barrier()

def init_distributed(rank: int, world_size: int, port: int = None):
    if not torch.cuda.is_available() or world_size == 1:
        return  # no DDP needed
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Set environment variables for DDP
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    
    # MASTER_PORT must already be set before spawning to avoid race conditions.
    # If not set, use provided port or fall back to a fixed default.
    if 'MASTER_PORT' not in os.environ:
        if port is None:
            port = 8972
        os.environ['MASTER_PORT'] = str(port)
    
    from datetime import timedelta
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )
def ddp_worker(rank: int, world_size: int, run_worker_fn, cfg, port: int = 12355, run_uuid=None):
    init_distributed(rank, world_size, port)
    try:
        run_worker_fn(rank, world_size, cfg, torch.device(f"cuda:{rank}"), run_uuid)
    finally:
        cleanup()

def cleanup():
     try:
         if dist.is_initialized():
             dist.destroy_process_group()
     except Exception as e:
         print(f"[WARNING] Error during cleanup: {e}")

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1

def barrier():
    if dist.is_initialized():
        dist.barrier()

def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

@torch.no_grad()
def broadcast_model(model: torch.nn.Module, src: int = 0):
    for p in model.parameters():
        dist.broadcast(p.data, src=src)
    for b in model.buffers():
        dist.broadcast(b.data, src=src)

def aggregate_metrics(metrics: dict, device: torch.device) -> dict:
    if not dist.is_initialized():
        return {k: float(v) for k, v in metrics.items()}
    tensors = {k: torch.tensor(v, device=device) for k, v in metrics.items()}
    for t in tensors.values():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: t.item() / get_world_size() for k, t in tensors.items()}

def wrap_ddp(model: torch.nn.Module, device: torch.device):
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])
    return model

__all__ = [
    "init_distributed", "cleanup", "main_process_first", "main_process_only", "is_main_process",
    "get_rank", "get_world_size", "barrier", "rank0_print", "broadcast_model",
    "aggregate_metrics", "ddp_worker", "wrap_ddp"
]
