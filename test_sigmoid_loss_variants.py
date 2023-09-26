import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from rwightman_sigmoid_loss import SigLipLoss
from distributed_sigmoid_loss import DDPSigmoidLoss


from test_distributed_sigmoid_loss import setup, set_seed
from test_distributed_sigmoid_loss import get_partition, get_encoders, average_gradients


def toy_rw_forward_backward_pass(rank, world_size, bz, emb_dim=2, return_dict=None):
    setup(rank, world_size)
    assert bz % world_size == 0
    gpu_batch_size = bz // world_size

    image_inputs, text_inputs = get_partition(rank, world_size, gpu_batch_size, emb_dim)
    image_encoder, text_encoder = get_encoders(emb_dim)

    # Toy forward (compute embedding)
    image_embeddings = image_encoder(image_inputs)
    text_embeddings = text_encoder(text_inputs)

    # L2 Normalize features
    image_embeddings = F.normalize(image_embeddings)
    text_embeddings = F.normalize(text_embeddings)

    # Compute loss
    logit_scale = nn.Parameter(torch.ones([]) * np.log(10))
    logit_bias = nn.Parameter(torch.ones([]) * -10)
    sigloss = SigLipLoss(rank=rank, world_size=world_size)
    loss = sigloss(image_embeddings, text_embeddings, logit_scale, logit_bias)

    # Toy backward (compute gradients)
    loss.backward()

    # average_gradients(text_encoder)
    # # # check gradients
    # print(f"Rank:{rank} text_encoder.weight.grad: {text_encoder.weight.grad}")

    # # # average gradient from all devices
    # average_gradients(image_encoder)
    # # # check gradients
    # print(f"Rank:{rank} image_encoder.weight.grad: {image_encoder.weight.grad}")

    if rank == 0:
        return_dict['img_grad'] = image_encoder.weight.grad
        return_dict['txt_grad'] = text_encoder.weight.grad


def toy_ddp_forward_backward_pass(rank, world_size, bz, emb_dim=2, return_dict=None):
    setup(rank, world_size)
    assert bz % world_size == 0
    gpu_batch_size = bz // world_size

    image_inputs, text_inputs = get_partition(rank, world_size, gpu_batch_size, emb_dim)
    image_encoder, text_encoder = get_encoders(emb_dim)

    # Toy forward (compute embedding)
    image_embeddings = image_encoder(image_inputs)
    text_embeddings = text_encoder(text_inputs)

    # L2 Normalize features
    image_embeddings = F.normalize(image_embeddings)
    text_embeddings = F.normalize(text_embeddings)

    # Compute loss
    loss = DDPSigmoidLoss(gpu_batch_size)(image_embeddings, text_embeddings)

    # Toy backward (compute gradients)
    loss.backward()

    # average_gradients(text_encoder)
    # # # check gradients
    # print(f"Rank:{rank} text_encoder.weight.grad: {text_encoder.weight.grad}")

    # # # average gradient from all devices
    # average_gradients(image_encoder)
    # # # check gradients
    # print(f"Rank:{rank} image_encoder.weight.grad: {image_encoder.weight.grad}")

    if rank == 0:
        return_dict['img_grad'] = image_encoder.weight.grad
        return_dict['txt_grad'] = text_encoder.weight.grad


def compare_naive_vs_rw(world_size=3, batch_size=3, emb_dim=2):
    manager = mp.Manager()
    rw_return_dict = manager.dict()

    mp.spawn(
        toy_rw_forward_backward_pass,
        args=(world_size, batch_size, emb_dim, rw_return_dict),
        nprocs=world_size,
        join=True,
    )

    ddp_return_dict = manager.dict()
    mp.spawn(
        toy_ddp_forward_backward_pass,
        args=(world_size, batch_size, emb_dim, ddp_return_dict),
        nprocs=world_size,
        join=True,
    )

    assert torch.allclose(rw_return_dict['img_grad'], ddp_return_dict['img_grad'], rtol=1e-3)
    assert torch.allclose(rw_return_dict['txt_grad'], ddp_return_dict['txt_grad'], rtol=1e-3)


if __name__ == '__main__':
    compare_naive_vs_rw()
    compare_naive_vs_rw(world_size=2, batch_size=4, emb_dim=4)
    compare_naive_vs_rw(world_size=2, batch_size=4, emb_dim=128)
