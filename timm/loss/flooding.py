""" Flooding loss

    From paper: Do We Need Zero Training Loss After Achieving Zero Training Error?
    https://arxiv.org/abs/2002.08709

Hacked together by / Copyright 2022 klae01
"""
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger("flooding")


class Flooding(nn.Module):
    def __init__(self, loss_module: nn.Module, flooding: float = None):
        super().__init__()
        self.loss_module = loss_module
        self.loss_state_dict = nn.ParameterDict(
            copy.deepcopy(self.loss_module).state_dict()
        )
        self.replicated_loss = copy.deepcopy(self.loss_module)
        self.flooding = flooding

    def lower_bound(
        self, x: torch.Tensor, target: torch.Tensor, steps: int = 35
    ) -> float:
        torch_opt = {"device": target.device}
        self.replicated_loss.load_state_dict(dict(self.loss_state_dict))
        X = torch.nn.Parameter(
            torch.randn(
                x.size(),
                generator=torch.Generator(x.device).manual_seed(0),
                **torch_opt,
            )
        )
        opt = torch.optim.Rprop([X], lr=1.0)
        min_loss = torch.tensor(torch.inf, **torch_opt)
        for _ in range(steps):
            opt.zero_grad()
            L = self.replicated_loss(X, target)
            L.mean().backward()
            opt.step()
            with torch.no_grad():
                min_loss = torch.minimum(min_loss, L)
        return min_loss

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.flooding is None:
            return self.loss_module(x, target).mean()

        loss = self.loss_module(x, target)
        assert (
            loss.ndim > 0
        ), f"Loss class {type(self.jit_loss).__name__} must return element-wise loss. (such as batch dimension)"
        flooding = self.lower_bound(x, target).to(loss.device) + self.flooding
        return abs(loss - flooding).mean() + flooding.mean()
