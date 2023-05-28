import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

import numpy as np
import math

EPSILON = 1e-6

from .utils import SparsityMetrics, SmoothStep


class COMET(nn.Module):
    """An ensemble of soft decision trees.

    The layer returns the sum of the decision trees in the ensemble.
    Each soft tree returns a vector, whose dimension is specified using
    the `leaf_dims' parameter.

    Implementation Notes:
        This is a fully vectorized implementation. It treats the ensemble
        as one "super" tree, where every node stores a dense layer with
        num_trees units, each corresponding to the hyperplane of one tree.

    Input:
        An input tensor of shape = (batch_size, ...)

    Output:
        An output tensor of shape = (batch_size, leaf_dims)
    """

    def __init__(self, config, node_index=0, depth_index=0, name="Node-Root"):
        """Initializes the layer.
        Args:
          input_dim: The dimension of the input tensor.
          num_trees: The number of trees in the ensemble.
          leaf_dims: The dimension of the output vector.
          gamma: The scaling parameter for the smooth-step function.
        """
        super(COMET, self).__init__()

        self.nb_experts = config["nb_experts"]
        self.max_depth = (int)(np.ceil(np.log2(self.nb_experts)))
        self.k = int(config["k"])

        self.node_index = node_index
        self.depth_index = depth_index
        self.max_split_nodes = self.nb_experts - 1
        self.leaf = node_index >= self.nb_experts - 1

        self.gamma = config["gamma"]

        self.activation = SmoothStep(self.gamma)

        self.input_dim = config["input_dim"]

        self.entropy_reg = config["entropy_reg"]

        self.balanced_splitting = config.get("balanced_splitting", False)
        self.balanced_splitting_penalty = config.get("balanced_splitting_penalty", 0.0)
        self.exp_decay_mov_ave = config.get("exp_decay_mov_ave", 0.0)

        self.sparsity_metrics = SparsityMetrics()

        if not self.leaf:
            self.selector_layer = nn.Linear(self.input_dim, self.k, bias=False)
            self.selector_layer.weight = self._z_initializer(self.selector_layer.weight)

            self.left_child = COMET(config, node_index=2 * node_index + 1, depth_index=depth_index + 1, name="Node-Left")
            self.right_child = COMET(config, node_index=2 * node_index + 2, depth_index=depth_index + 1, name="Node-Right")
            if self.balanced_splitting:
                self.alpha_ave_past_tensor = torch.tensor(0.0, dtype=torch.float32)
                self.alpha_ave_past = nn.Parameter(self.alpha_ave_past_tensor, requires_grad=False)

        else:
            self.output_layer = nn.Linear(self.input_dim, self.k)
            self.output_layer.weight = self._w_initializer(self.output_layer.weight)
            self.output_layer.bias.data.fill_(0.0)

        if self.node_index == 0:
            self.permutation_mask = torch.tensor(np.array([np.identity(self.nb_experts) for _ in range(self.k)]), dtype=torch.float32)

    def _z_initializer(self, x):
        return nn.init.uniform_(x, -self.gamma / 100, self.gamma / 100)

    def _w_initializer(self, x):
        return nn.init.uniform_(x, a=-0.05, b=0.05)

    def _compute_balanced_split_loss(self, prob, current_prob):
        exp_decay_mov_ave = self.exp_decay_mov_ave * (1.0**self.depth_index)

        penalty = self.balanced_splitting_penalty / (1.0**self.depth_index)

        alpha_ave_current = torch.sum(prob * current_prob, dim=0) / (torch.sum(prob * torch.ones_like(current_prob), dim=0) + EPSILON)
        alpha_ave = (1 - exp_decay_mov_ave) * alpha_ave_current.float() + exp_decay_mov_ave * self.alpha_ave_past.float()

        self.alpha_ave_past = nn.Parameter(alpha_ave, requires_grad=False)

        loss = -penalty * (0.5 * torch.log(alpha_ave + EPSILON) + 0.5 * torch.nn.math.log(1 - alpha_ave + EPSILON))
        loss = torch.mean(loss, dim=-1)
        loss = torch.sum(loss)
        return loss

    def _compute_entropy_rdegularization_per_expert(self, prob, entropy_reg):
        prob = torch.clamp(prob, min=EPSILON)

        regularization = entropy_reg * torch.mean(torch.sum(-(prob + EPSILON) * torch.log(prob + EPSILON), dim=1))

        return regularization

    def forward(self, inputs, training=True, prob=1.0):
        regularization_loss = 0.0

        h, x = inputs

        assert all([h[i].shape[1] == h[i + 1].shape[1] for i in range(len(h) - 1)])

        h = [torch.unsqueeze(t, -1) for t in h]

        h = torch.concat(h, dim=2)


        if not self.leaf:
            
            current_prob = self.selector_layer(x)  # (batch_size, k)
            current_prob = self.activation.forward(current_prob)  # (batch_size, k)

            s_left_child, regularization_loss_left = self.left_child.forward(inputs, training=training, prob=current_prob * prob)
            s_right_child, regularization_loss_right = self.right_child.forward(inputs, training=training, prob=(1 - current_prob) * prob)

            regularization_loss = regularization_loss_left + regularization_loss_right


            s_bj = torch.cat([s_left_child, s_right_child], dim=2)

            if self.node_index == 0:  # root node
               
                h = torch.unsqueeze(h, dim=2)

                s_bj = torch.reshape(s_bj, shape=[s_bj.shape[0], -1])  # (b, k*nb_experts)
                s_bj = torch.softmax(s_bj, dim=-1)  # (b, k*nb_experts)



                w_concat = torch.reshape(s_bj, shape=[s_bj.shape[0], self.k, self.nb_experts])  # (b, k, nb_experts)
                w_concat = torch.unsqueeze(w_concat, dim=1)  # (b, 1, k, nb_experts)

                # w_concat: (b, 1, k, nb_experts), perm_mask: [k, nb_experts, nb_experts]
                w_permuted = torch.einsum("bijk,jkl->bijl", w_concat, self.permutation_mask.to(w_concat.device))
                w_permuted = torch.sum(w_permuted, dim=2, keepdim=True)  # (b, 1, 1, nb_experts)
                w_permuted = w_permuted / torch.sum(w_permuted, dim=-1, keepdim=True)  # (b, 1, 1, nb_experts)

                # h:(b, dim_exp_i, 1, nb_experts) * w_permuted: (b, 1, 1, nb_experts)
                y_agg = torch.sum(h * w_permuted, dim=[2, 3])  # (b, dim_exp_i, 1, nb_experts) -> (b, dim_exp_i)

                s_concat = torch.where(
                    torch.less(w_permuted, 1e-5), torch.ones_like(w_permuted), torch.zeros_like(w_permuted)
                )  # (b, 1, 1, nb_experts)
                s_avg = torch.mean(s_concat, dim=-1)  # (b, 1, 1)


                if self.training:
                    avg_sparsity = torch.mean(s_avg)  # average over batch
                    self.sparsity_metrics.update(avg_sparsity)

                soft_averages = torch.mean(w_permuted, dim=[0, 1, 2])  # (nb_experts,)
                hard_averages = torch.mean(torch.ones_like(s_concat) - s_concat, dim=[0, 1, 2])  # (nb_experts,)

                return (
                    y_agg,
                    soft_averages,
                    hard_averages,
                    s_concat,
                    regularization_loss,
                )  # For root node, return the aggregated output and the sparsity of the weights, and the sparsity of the weights for each expert
            else:
                return s_bj, regularization_loss
        else:

            # a_bij shape = (b, k)
            a_bij = self.output_layer(x)  # (b, k) # j is a leave, no access

            prob = torch.unsqueeze(prob, dim=-1)  # (b, k, 1)
            a_bij = torch.unsqueeze(a_bij, dim=-1)  # (b, k, 1)

            log_prob = torch.where(prob < 1e-5, -torch.ones_like(prob) * torch.inf, torch.log(prob + +1e-6))

            s_bj = a_bij + log_prob  # (b, k, 1)

            if training:
                regularization_loss = self._compute_entropy_rdegularization_per_expert(prob, entropy_reg=self.entropy_reg)
            else:
                regularization_loss = 0.0

            return s_bj, regularization_loss 

