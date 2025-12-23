import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Type, Any

class ShardedOptimizer(Optimizer):

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.all_params = []  # (param, owner_rank)
        
        # Normalize params to list
        params_list = [params] if isinstance(params, torch.Tensor) else list(params)
        
        # Shard parameters among ranks
        sharded_params = []
        for i, p in enumerate(params_list):
            owner_rank = i % self.world_size
            self.all_params.append((p, owner_rank))
            if owner_rank == self.rank:
                sharded_params.append(p)
        
        # Initialize wrapped optimizer
        self.optimizer = optimizer_cls(sharded_params, **kwargs) if sharded_params else None
        
        # Initialize parent
        super().__init__([{'params': []}], kwargs)
        self.param_groups.clear()
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure) if self.optimizer else None
        
        # Broadcast updated params from owners
        for param, owner_rank in self.all_params:
            dist.broadcast(param.data, src=owner_rank)
        
        return loss
    
    def zero_grad(self, set_to_none=False):
        if self.optimizer:
            self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def add_param_group(self, param_group: dict[str, Any]):
        sharded_params = []
        for i, p in enumerate(param_group['params']):
            owner_rank = (len(self.all_params) + i) % self.world_size
            self.all_params.append((p, owner_rank))
            if owner_rank == self.rank:
                sharded_params.append(p)
        
        if sharded_params and self.optimizer:
            new_group = {k: v for k, v in param_group.items() if k != 'params'}
            new_group['params'] = sharded_params
            self.optimizer.add_param_group(new_group)