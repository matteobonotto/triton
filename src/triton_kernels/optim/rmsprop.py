from typing import Optional
from torch.optim import RMSprop as RMSpropTorch
from torch import Tensor
import torch
from typing import Union, Optional

# from .ops.rmsprop import rmsprop
from .ops.rmsprop import rmsprop

# @torch.no_grad()
# def rmsprop(
#     params: list[Tensor],
#     grads: list[Tensor],
#     square_avgs: list[Tensor],
#     state_steps: list[Tensor],
#     lr: float = 1e-2,
#     alpha: float = 0.99,
#     eps: float = 1e-8,
# ):

#     for i, param in enumerate(params):
#         step = state_steps[i]
#         grad = grads[i]
#         square_avg = square_avgs[i]

#         step += 1
#         # don't need to output, just update the various params inside the for loop

#         square_avg *= alpha
#         square_avg += (1 - alpha) * grad.pow(2)  # <- optimizer params, shape (N, )
#         param -= lr * grad / (torch.sqrt(square_avg) + eps)
#     return


class RMSprop(RMSpropTorch):  # noqa: D101
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, **kwargs) -> Optional[Tensor]:
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # self._cuda_graph_capture_health_check()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            square_avgs: list[Tensor] = []
            grad_avgs: list[Tensor] = []
            momentum_buffer_list: list[Tensor] = []
            state_steps: list[Tensor] = []

            _ = self._init_group(
                group,
                params_with_grad,
                grads,
                square_avgs,
                momentum_buffer_list,
                grad_avgs,
                state_steps,
            )

            rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                # grad_avgs,
                # momentum_buffer_list,
                state_steps,
                lr=group["lr"],
                alpha=group["alpha"],
                eps=group["eps"],
                # weight_decay=group["weight_decay"],
                # momentum=group["momentum"],
                # centered=group["centered"],
                # foreach=group["foreach"],
                # maximize=group["maximize"],
                # differentiable=group["differentiable"],
                # capturable=group["capturable"],
                # has_complex=has_complex,
            )

        return None
