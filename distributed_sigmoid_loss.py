import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


class DDPSigmoidLoss(nn.Module):
    def __init__(self, gpu_batch_size) -> None:
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor(np.log(1 / 0.07)))  # log 10
        self.bias = nn.Parameter(torch.tensor(-10.0))

        self.gpu_batch_size = gpu_batch_size
        self.sigmoid_loss = nn.LogSigmoid()

    def forward(self, image_embeddings, text_embeddings):
        # image_embeddings: (gpu_batch_size, emb_dim)
        # text_embeddings: (world_size * gpu_batch_size, emb_dim)
        # All embeddings as assumes to be L2 normalized

        def compute_device_loss(zimg, ztxt, same_device):
            t = self.t_prime.exp()
            logits = zimg @ ztxt.T * t + self.bias

            if same_device:
                # -1 with diagonal 1
                labels = 2 * torch.eye(self.gpu_batch_size) - torch.ones(self.gpu_batch_size)
            else:
                labels = -1 * torch.ones(self.gpu_batch_size)  # -1

            loss = -self.sigmoid_loss(labels * logits)
            return loss.sum()

        all_text_embeddings = dist_nn.all_gather(text_embeddings)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        total_loss = 0
        for i in range(world_size):
            gpu_batch_loss = compute_device_loss(
                image_embeddings, all_text_embeddings[i], same_device=i == rank
            )
            total_loss += gpu_batch_loss

        total_loss = total_loss / self.gpu_batch_size
        return total_loss
