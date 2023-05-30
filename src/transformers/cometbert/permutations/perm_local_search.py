import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import gc


class PermLocalSearch(nn.Module):
    """Learn permutations.
    Input:
        An input tensor of shape = (batch_size, ...)
    Output:
        An output tensor of shape = (1, nb_experts, nb_experts)
    """

    def __init__(self, config):
        super(PermLocalSearch, self).__init__()
        self.nb_experts = config["nb_experts"]
        self.k = config["k"]

        self.tau_initial = 1e-3
        self.tau_final = 1e-7
        self.steps_per_epoch = config["steps_per_epoch"]
        self.epochs_for_learning_permutation = config["epochs_for_learning_permutation"]
        self.iterations_for_learning_permutation = self.steps_per_epoch * self.epochs_for_learning_permutation
        self.tau_ref = torch.tensor(np.linspace(np.log10(self.tau_initial), np.log10(self.tau_final), num=2), dtype=torch.float32)
        self.n_iters_ref = torch.tensor(np.linspace(20, 150, num=2), dtype=torch.float32)

        self.learn_k_permutations = config["learn_k_permutations"]
        if self.learn_k_permutations:
            self.no_of_permutations = self.k
        else:
            self.no_of_permutations = 1

        self.noise_factor = config["noise_factor"]
        self.perm_entropy_reg = config["perm_entropy_reg"]

        self.iterations = torch.tensor(0, dtype=torch.int32)

        self.permutation_weights = torch.nn.Parameter(
            torch.tensor(np.array([np.zeros(shape=(self.nb_experts, self.nb_experts)) for _ in range(self.no_of_permutations)]), dtype=torch.float32)
        )
        self.permutation_weights.requires_grad = False

        self.permutation_log_weights = torch.nn.Parameter(
            torch.tensor(
                np.array([np.random.uniform(size=(self.nb_experts, self.nb_experts), low=-0.05, high=0.05) for _ in range(self.no_of_permutations)]),
                dtype=torch.float32,
            )
        )
        self.permutation_log_weights.requires_grad = True
        self.permutation_log_weights.retain_grad()

    def _sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def _sinkhorn(self, log_alpha, n_iter=20):
        b = log_alpha.shape[0]
        log_alpha = torch.reshape(log_alpha, (b, self.nb_experts, self.nb_experts))
        # gc.collect()
        for _ in range(n_iter):
            log_alpha = log_alpha - torch.unsqueeze(torch.logsumexp(log_alpha, dim=2), dim=2)
            log_alpha = log_alpha - torch.unsqueeze(torch.logsumexp(log_alpha, dim=1), dim=1)
        return torch.exp(log_alpha)

    def _gumbel_sinkhorn(self, log_alpha, temp=1.0, n_samples=1, noise_factor=0.0, n_iters=20, squeeze=True):
        n = log_alpha.shape[1]
        batch_size = log_alpha.shape[0]
        log_alpha = torch.reshape(log_alpha, (batch_size, n, n))
        log_alpha_w_noise = torch.tile(log_alpha, (n_samples, 1, 1))
        if noise_factor == 0.0:
            noise = torch.tensor(0.0, dtype=torch.float32)
        else:
            noise = self._sample_gumbel([n_samples * batch_size, n, n]) * noise_factor
        noise = noise.to(log_alpha.device)
        log_alpha_w_noise += noise
        log_alpha_w_noise /= temp
        sink = self._sinkhorn(log_alpha_w_noise, n_iters)

        return sink

    def _generate_mask_per_permutation(self, permutation_weights):
        permutation_weights = torch.squeeze(permutation_weights)
        cost = -permutation_weights
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost.cpu().detach().numpy())  # to recheck
        permutation_mask = torch.eye(permutation_weights.shape[-1])[torch.tensor(col_ind)]
        return permutation_mask

    def _get_permutation_mask(self, permutation_weights):
        permutation_weights_list = torch.split(permutation_weights, self.no_of_permutations)
        permutation_masks_list = [self._generate_mask_per_permutation(perm) for perm in permutation_weights_list]
        permutation_masks = torch.stack(permutation_masks_list).to(permutation_weights.device)
        return permutation_masks

    def _compute_permutation_entropy_regularization(self, permutation_weights, eps=1e-6):
        permutation_weights_row_norm = permutation_weights / torch.sum(permutation_weights, dim=-1, keepdim=True)

        regularization = torch.mean(torch.sum(-permutation_weights * torch.log(permutation_weights + eps), dim=(1, 2))) + torch.mean(
            torch.sum(-permutation_weights_row_norm * torch.log(permutation_weights_row_norm + eps), dim=(1, 2))
        )

        return regularization

    def _get_permutation_during_training(self, permutation_log_weights, noise_factor=0.0):
        log_tau = (
            np.log10(self.tau_initial)
            + (np.log10(self.tau_final) - np.log10(self.tau_initial)) * self.iterations / self.iterations_for_learning_permutation
        )  # gives the same result as the one with tf version in the original code

        tau = torch.pow(10.0, log_tau)

        # stop the gradient from flowing back to the permutation weights if iterations are greater than the number of iterations for learning the permutation`

        permutation_log_weights = torch.where(
            self.iterations > self.iterations_for_learning_permutation,
            permutation_log_weights.detach(),  # .clone() as well ??
            permutation_log_weights,
        )

        # permutation_log_weights = tf.cond(
        #    tf.math.greater_equal(self.iterations, tf.cast(self.iterations_for_learning_permutation, dtype=self.iterations.dtype)),
        #    lambda: tf.stop_gradient(permutation_log_weights),
        #    lambda: permutation_log_weights
        # )

        n_iters = torch.tensor(20 + (150 - 20) * self.iterations / self.iterations_for_learning_permutation)  # Now it's a tensor and checked

        n_iters = torch.clamp(n_iters, min=20, max=150)
        n_iters = n_iters.long()
        permutation_weights = self._gumbel_sinkhorn(
            permutation_log_weights, temp=tau, n_samples=1, noise_factor=noise_factor, n_iters=n_iters, squeeze=True
        )

        self.permutation_weights = nn.Parameter(permutation_weights, requires_grad=False)

        return permutation_weights

    def _get_permutation_during_inference(self, permutation_weights):
        permutation_weights = self._get_permutation_mask(permutation_weights)
        #         tf.print("====iteration:", self.iterations, "==perm", permutation_weights)
        return permutation_weights

    def _get_permutation_during_learning_and_after_learning(self, iterations):
        # norm = torch.linalg.norm(
        #    self._get_permutation_during_inference(self.permutation_weights) - self.permutation_weights, dim=(1, 2)
        # )

        # print(self._get_permutation_during_inference(self.permutation_weights).device)

        permutation_weights = torch.where(
            torch.less(
                iterations.to(self.permutation_weights.device),
                torch.tensor(self.iterations_for_learning_permutation, dtype=self.iterations.dtype).to(self.permutation_weights.device),
            ),
            self._get_permutation_during_training(self.permutation_log_weights, noise_factor=self.noise_factor),
            self._get_permutation_during_inference(self.permutation_weights),
        )
        return permutation_weights

    def forward(self, inputs):
        training = self.training

        self.iterations = self.iterations.to(inputs.device)
        if training:
            increment = torch.ones_like(self.iterations)
        else:
            increment = torch.zeros_like(self.iterations)
        self.iterations += increment

        if training:
            permutation_weights = self._get_permutation_during_learning_and_after_learning(self.iterations)
        else:
            permutation_weights = self._get_permutation_during_inference(self.permutation_weights)

        if training:
            explicit_perm_entropy_regularization = self._compute_permutation_entropy_regularization(permutation_weights)
        else:
            explicit_perm_entropy_regularization = torch.zeros_like(torch.empty(1))

        if not self.learn_k_permutations:
            permutation_weights = torch.tile(permutation_weights, torch.tensor([self.k, 1, 1]))

        return permutation_weights, self.perm_entropy_reg * explicit_perm_entropy_regularization
