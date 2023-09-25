import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


from distributed_sigmoid_loss import DDPSigmoidLoss


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2 ** 32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup(rank, world_size):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "file:///{your local file path}"

        # initialize the process group
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_partition(rank, world_size, gpu_batch_size, emb_dim):
    set_seed(42)
    # image_inputs = torch.range(0, bz * emb_dim - 1).reshape(bz, emb_dim)
    # set_seed(42)
    # text_inputs = torch.range(0, bz * emb_dim - 1).reshape(bz, emb_dim)
    image_inputs = torch.randn(world_size * gpu_batch_size, emb_dim)
    set_seed(40)
    text_inputs = torch.randn(world_size * gpu_batch_size, emb_dim)
    return (
        image_inputs[rank * gpu_batch_size : (rank + 1) * gpu_batch_size],
        text_inputs[rank * gpu_batch_size : (rank + 1) * gpu_batch_size],
    )


def get_encoders(emb_dim, output_dim=2):
    set_seed(42)
    image_encoder = nn.Linear(emb_dim, output_dim, bias=False)
    set_seed(42)
    text_encoder = nn.Linear(emb_dim, output_dim, bias=False)
    return image_encoder, text_encoder


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def toy_forward_backward_pass(rank, world_size, bz, emb_dim=2, return_dict=None):

    setup(rank, world_size)
    assert bz % world_size == 0
    gpu_batch_size = bz // world_size

    image_inputs, text_inputs = get_partition(rank, world_size, gpu_batch_size, emb_dim)
    image_encoder, text_encoder = get_encoders(emb_dim)

    # Toy forward (compute embedding)
    image_embeddings = image_encoder(image_inputs)
    text_embeddings = text_encoder(text_inputs)

    # Compute loss
    loss = DDPSigmoidLoss(gpu_batch_size)(image_embeddings, text_embeddings)

    # Toy backward (compute gradients)
    loss.backward()

    average_gradients(text_encoder)
    # # # check gradients
    # print(f"Rank:{rank} text_encoder.weight.grad: {text_encoder.weight.grad}")

    # # # average gradient from all devices
    average_gradients(image_encoder)
    # # # check gradients
    # print(f"Rank:{rank} image_encoder.weight.grad: {image_encoder.weight.grad}")
    if rank == 0:
        return_dict['img_grad'] = image_encoder.weight.grad
        return_dict['txt_grad'] = text_encoder.weight.grad


def test_odd_world_size(emb_dim=2):
    manager = mp.Manager()
    world_size, batch_size = 3, 3
    w3_return_dict = manager.dict()
    mp.spawn(
        toy_forward_backward_pass,
        args=(world_size, batch_size, emb_dim, w3_return_dict),
        nprocs=world_size,
        join=True,
    )

    w1_return_dict = manager.dict()
    world_size, batch_size = 1, 3
    mp.spawn(
        toy_forward_backward_pass,
        args=(world_size, batch_size, emb_dim, w1_return_dict),
        nprocs=world_size,
        join=True,
    )

    assert torch.allclose(w3_return_dict['img_grad'], w1_return_dict['img_grad'])
    assert torch.allclose(w3_return_dict['txt_grad'], w1_return_dict['txt_grad'])


def test_even_world_size(emb_dim=2):
    manager = mp.Manager()
    world_size, batch_size = 2, 4
    w2_return_dict = manager.dict()
    mp.spawn(
        toy_forward_backward_pass,
        args=(world_size, batch_size, emb_dim, w2_return_dict),
        nprocs=world_size,
        join=True,
    )

    w1_return_dict = manager.dict()
    world_size, batch_size = 1, 4
    mp.spawn(
        toy_forward_backward_pass,
        args=(world_size, batch_size, emb_dim, w1_return_dict),
        nprocs=world_size,
        join=True,
    )
    # print(w2_return_dict['img_grad'])
    # print(w1_return_dict['img_grad'])
    assert torch.allclose(w2_return_dict['img_grad'], w1_return_dict['img_grad'], rtol=1e-4)
    assert torch.allclose(w2_return_dict['txt_grad'], w1_return_dict['txt_grad'], rtol=1e-4)


if __name__ == "__main__":
    # test_odd_world_size()
    # test_even_world_size()
    test_even_world_size(emb_dim=128)
    test_even_world_size(emb_dim=512)
