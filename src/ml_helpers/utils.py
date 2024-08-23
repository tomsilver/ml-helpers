"""Utility functions."""

import os
import tempfile
from typing import Callable, Iterator, Tuple

import numpy as np
import torch
from torch import Tensor, nn, optim

from ml_helpers.structs import Array, MaxTrainIters

torch.use_deterministic_algorithms(mode=True)  # type: ignore


def _get_torch_device(use_torch_gpu: bool) -> torch.device:
    return torch.device(
        "cuda:0" if use_torch_gpu and torch.cuda.is_available() else "cpu"
    )


def _normalize_data(data: Array, scale_clip: float = 1) -> Tuple[Array, Array, Array]:
    shift = np.min(data, axis=0)
    scale = np.max(data - shift, axis=0)
    scale = np.clip(scale, scale_clip, None)
    return (data - shift) / scale, shift, scale


def _balance_binary_classification_data(
    X: Array, y: Array, rng: np.random.Generator
) -> Tuple[Array, Array]:
    pos_idxs_np = np.argwhere(np.array(y) == 1).squeeze()
    neg_idxs_np = np.argwhere(np.array(y) == 0).squeeze()
    pos_idxs = [pos_idxs_np.item()] if not pos_idxs_np.shape else list(pos_idxs_np)
    neg_idxs = [neg_idxs_np.item()] if not neg_idxs_np.shape else list(neg_idxs_np)
    assert len(pos_idxs) + len(neg_idxs) == len(y) == len(X)
    keep_neg_idxs = list(rng.choice(neg_idxs, replace=False, size=len(pos_idxs)))
    keep_idxs = pos_idxs + keep_neg_idxs
    X_lst = [X[i] for i in keep_idxs]
    y_lst = [y[i] for i in keep_idxs]
    X = np.array(X_lst)
    y = np.array(y_lst)
    return (X, y)


def _single_batch_generator(
    tensor_X: Tensor, tensor_Y: Tensor
) -> Iterator[Tuple[Tensor, Tensor]]:
    """Infinitely generate all of the data in one batch."""
    while True:
        yield (tensor_X, tensor_Y)


def _train_pytorch_model(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: optim.Optimizer,
    batch_generator: Iterator[Tuple[Tensor, Tensor]],
    max_train_iters: MaxTrainIters,
    dataset_size: int,
    device: torch.device,
    print_every: int = 1000,
    clip_gradients: bool = False,
    clip_value: float = 5,
    n_iter_no_change: int = 10000000,
) -> float:
    """Note that this currently does not use minibatches.

    In the future, with very large datasets, we would want to switch to
    minibatches. Returns the best loss seen during training.
    """
    model.train()
    itr = 0
    best_loss = float("inf")
    best_itr = 0
    model_name = tempfile.NamedTemporaryFile(delete=False).name
    if isinstance(max_train_iters, int):
        max_iters = max_train_iters
    else:  # assume that it's a function from dataset size to max iters
        max_iters = max_train_iters(dataset_size)
    assert isinstance(max_iters, int)
    for tensor_X, tensor_Y in batch_generator:
        Y_hat = model(tensor_X)
        loss = loss_fn(Y_hat, tensor_Y)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_itr = itr
            # Save this best model.
            torch.save(model.state_dict(), model_name)
        if itr % print_every == 0:
            print(f"Loss: {loss:.5f}, iter: {itr}/{max_iters}")
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        if itr - best_itr > n_iter_no_change:
            print(
                f"Loss did not improve after {n_iter_no_change} "
                f"itrs, terminating at itr {itr}."
            )
            break
        if itr == max_iters:
            break
        itr += 1
    # Load best model.
    model.load_state_dict(torch.load(model_name, map_location="cpu"))  # type: ignore
    model.to(device)
    os.remove(model_name)
    model.eval()
    print(f"Loaded best model with loss: {best_loss:.5f}")
    return best_loss
