import numpy as np

import torch


class NoPermutations(torch.nn.Module):
    """No permutations (identity).

    Input:
        An input tensor of shape = (batch_size, ...) # not used

    Output:
        An output tensor of shape = (1, nb_experts, nb_experts)
    """

    def __init__(
        self,
        config,
    ):
        super(NoPermutations, self).__init__()
        self.nb_experts = config["nb_experts"]

        self.no_of_permutations = config["k"]

        self.permutation_weights = torch.tensor(
            np.array([np.identity(self.nb_experts) for _ in range(self.no_of_permutations)]), dtype=torch.float32
        )

    def forward(self, inputs):
        x = inputs

        """ trace_RRT = torch.trace(
            torch.matmul(self.permutation_weights, torch.transpose(self.permutation_weights, 1, 2))
        )
        trace_RTR = torch.trace(
            torch.matmul(torch.transpose(self.permutation_weights, 1, 2), self.permutation_weights)
        )
        #             self.add_metric(torch.mean(trace_RRT), name
        print("=========trace_RRT:", trace_RRT)
        print("=========trace_RTR:", trace_RTR)"""

        return self.permutation_weights.to(x.device), 0
