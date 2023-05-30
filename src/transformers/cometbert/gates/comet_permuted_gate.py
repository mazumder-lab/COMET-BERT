import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

import numpy as np
import math

EPSILON = 1e-6

from .utils import SparsityMetrics, SmoothStep


class COMETPermLocalSearch(nn.Module):
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
        super(COMETPermLocalSearch, self).__init__()

        self.nb_experts = config["nb_experts"]
        self.max_depth = (int)(np.ceil(np.log2(self.nb_experts)))
        self.k = config["k"]

        #         #print("=========self.nb_experts:", self.nb_experts)
        #         #print("=========self.max_depth:", self.max_depth)
        self.node_index = node_index
        #         #print("=========self.node_index:", self.node_index)
        self.depth_index = depth_index
        #         self.max_split_nodes = 2**self.max_depth - 1
        self.max_split_nodes = self.nb_experts - 1
        #         self.leaf = node_index >= self.max_split_nodes
        self.leaf = node_index >= self.nb_experts - 1
        #         assert self.nb_experts == 2**self.max_depth # to check number of experts is a power of 2

        self.gamma = config["gamma"]

        self.activation = SmoothStep(self.gamma)

        self.input_dim = config["input_dim"]

        self.entropy_reg = config["entropy_reg"]

        self.sparsity_metrics = SparsityMetrics()

        if not self.leaf:
            self.selector_layer = nn.Linear(self.input_dim, self.k, bias=False)
            self.selector_layer.weight = self._z_initializer(self.selector_layer.weight)

            self.left_child = COMETPermLocalSearch(config, node_index=2 * node_index + 1, depth_index=depth_index + 1, name="Node-Left")
            self.right_child = COMETPermLocalSearch(config, node_index=2 * node_index + 2, depth_index=depth_index + 1, name="Node-Right")

        else:
            self.output_layer = nn.Linear(self.input_dim, self.k)
            self.output_layer.weight = self._w_initializer(self.output_layer.weight)
            self.output_layer.bias.data.fill_(0.0)

    def _z_initializer(self, x):
        return nn.init.uniform_(x, -self.gamma / 100, self.gamma / 100)

    def _w_initializer(self, x):
        return nn.init.uniform_(x, a=-0.05, b=0.05)

    def _compute_entropy_regularization_per_expert(self, prob, entropy_reg):
        # clip values of prob to avoid nan
        prob = torch.clamp(prob, min=EPSILON)

        regularization = entropy_reg * torch.mean(torch.sum(-(prob + EPSILON) * torch.log(prob + EPSILON), dim=1))

        return regularization

    def forward(self, inputs, training=True, prob=1.0):
        regularization_loss = 0.0

        h, x, permutation_weights = inputs

        assert all([h[i].shape[1] == h[i + 1].shape[1] for i in range(len(h) - 1)])

        h = [torch.unsqueeze(t, -1) for t in h]

        h = torch.concat(h, dim=2)


        if not self.leaf:
           
            current_prob = self.selector_layer(x) 
            current_prob = self.activation.forward(current_prob) 

            s_left_child, regularization_loss_left = self.left_child.forward(inputs, training=training, prob=current_prob * prob)
            s_right_child, regularization_loss_right = self.right_child.forward(inputs, training=training, prob=(1 - current_prob) * prob)

            regularization_loss = regularization_loss_left + regularization_loss_right

            s_bj = torch.cat([s_left_child, s_right_child], dim=2)

            if self.node_index == 0:  # root node
                
                h = torch.unsqueeze(h, dim=2)

                s_bj = torch.reshape(s_bj, shape=[s_bj.shape[0], -1])  
                s_bj = torch.softmax(s_bj, dim=-1)  


                w_concat = torch.reshape(s_bj, shape=[s_bj.shape[0], self.k, self.nb_experts])  
                w_concat = torch.unsqueeze(w_concat, dim=1)



                w_permuted = torch.einsum("bijk,jkl->bijl", w_concat, permutation_weights)
                w_permuted = torch.sum(w_permuted, dim=2, keepdim=True)
                w_permuted = w_permuted / torch.sum(w_permuted, dim=-1, keepdim=True)

                y_agg = torch.sum(h * w_permuted, dim=[2, 3])

                s_concat = torch.where(
                    torch.less(w_permuted, 1e-5), torch.ones_like(w_permuted), torch.zeros_like(w_permuted)
                )
                s_avg = torch.mean(s_concat, dim=-1)


                if self.training:
                    avg_sparsity = torch.mean(s_avg) 
                    self.sparsity_metrics.update(avg_sparsity)

                soft_averages = torch.mean(w_permuted, dim=[0, 1, 2])
                hard_averages = torch.mean(torch.ones_like(s_concat) - s_concat, dim=[0, 1, 2])

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
            a_bij = self.output_layer(x) 

            prob = torch.unsqueeze(prob, dim=-1)
            a_bij = torch.unsqueeze(a_bij, dim=-1)

            log_prob = torch.where(prob < 1e-5, -torch.ones_like(prob) * torch.inf, torch.log(prob + +1e-6))
            s_bj = a_bij + log_prob

            if training:
                regularization_loss = self._compute_entropy_regularization_per_expert(prob, entropy_reg=self.entropy_reg)
            else:
                regularization_loss = 0.0

            return s_bj, regularization_loss  
