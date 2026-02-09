from typing import Optional
import torch
from torch import Tensor
from torch.optim import Adam as AdamTorch

from .ops.adam import adam

"""
def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
):
"""


class Adam(AdamTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, **kwargs) -> Optional[Tensor]:
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                # max_exp_avg_sqs,
                state_steps,
                # amsgrad=group["amsgrad"],
                # has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                # weight_decay=group["weight_decay"],
                eps=group["eps"],
                # maximize=group["maximize"],
                # foreach=group["foreach"],
                # capturable=group["capturable"],
                # differentiable=group["differentiable"],
                # fused=group["fused"],
                # grad_scale=getattr(self, "grad_scale", None),
                # found_inf=getattr(self, "found_inf", None),
                # decoupled_weight_decay=group["decoupled_weight_decay"],
            )

        return
