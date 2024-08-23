"""Models for regression."""

import abc
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor as _SKLearnKNeighborsRegressor
from torch import Tensor, nn, optim
from torch.distributions.categorical import Categorical

from ml_helpers.structs import Array, MaxTrainIters
from ml_helpers.utils import (
    get_torch_device,
    normalize_data,
    single_batch_generator,
    train_pytorch_model,
)


class Regressor(abc.ABC):
    """ABC for regressor classes."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    @abc.abstractmethod
    def fit(self, X: Array, Y: Array) -> None:
        """Train the regressor on the given data.

        X and Y are both two-dimensional.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def predict(self, x: Array) -> Array:
        """Return a prediction for the given datapoint.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me!")


class _ScikitLearnRegressor(Regressor):
    """A regressor that lightly wraps a scikit-learn regression model."""

    def __init__(self, seed: int, **kwargs: Any) -> None:
        super().__init__(seed)
        self._model = self._initialize_model(**kwargs)

    @abc.abstractmethod
    def _initialize_model(self, **kwargs: Any) -> BaseEstimator:
        raise NotImplementedError("Override me!")

    def fit(self, X: Array, Y: Array) -> None:
        return self._model.fit(X, Y)  # type: ignore

    def predict(self, x: Array) -> Array:
        return self._model.predict([x])[0]  # type: ignore


class _NormalizingRegressor(Regressor):
    """A regressor that normalizes the data.

    Also infers the dimensionality of the inputs and outputs from fit().
    """

    def __init__(self, seed: int, disable_normalization: bool = False) -> None:
        super().__init__(seed)
        # Set in fit().
        self._x_dims: tuple[int, ...] = tuple()
        self._y_dim = -1
        self._disable_normalization = disable_normalization
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._output_shift = np.zeros(1, dtype=np.float32)
        self._output_scale = np.zeros(1, dtype=np.float32)

    def fit(self, X: Array, Y: Array) -> None:
        num_data = X.shape[0]
        self._x_dims = tuple(X.shape[1:])
        _, self._y_dim = Y.shape
        assert Y.shape[0] == num_data
        print(f"Training {self.__class__.__name__} on {num_data} " "datapoints")
        if not self._disable_normalization:
            X, self._input_shift, self._input_scale = normalize_data(X)
            Y, self._output_shift, self._output_scale = normalize_data(Y)
        self._fit(X, Y)

    def predict(self, x: Array) -> Array:
        assert len(self._x_dims), "Fit must be called before predict."
        assert x.shape == self._x_dims
        # Normalize.
        if not self._disable_normalization:
            x = (x - self._input_shift) / self._input_scale
        # Make prediction.
        y = self._predict(x)
        assert y.shape == (self._y_dim,)
        # Denormalize.
        if not self._disable_normalization:
            y = (y * self._output_scale) + self._output_shift
        return y

    @abc.abstractmethod
    def _fit(self, X: Array, Y: Array) -> None:
        """Train the regressor on normalized data."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _predict(self, x: Array) -> Array:
        """Return a normalized prediction for the normalized input."""
        raise NotImplementedError("Override me!")


class PyTorchRegressor(_NormalizingRegressor, nn.Module):
    """ABC for PyTorch regression models."""

    def __init__(
        self,
        seed: int,
        max_train_iters: MaxTrainIters,
        clip_gradients: bool,
        clip_value: float,
        learning_rate: float,
        weight_decay: float = 0,
        n_iter_no_change: int = 10000000,
        use_torch_gpu: bool = False,
        train_print_every: int = 1000,
        disable_normalization: bool = False,
    ) -> None:
        torch.manual_seed(seed)
        _NormalizingRegressor.__init__(
            self, seed, disable_normalization=disable_normalization
        )
        nn.Module.__init__(self)  # type: ignore
        self._max_train_iters = max_train_iters
        self._clip_gradients = clip_gradients
        self._clip_value = clip_value
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._n_iter_no_change = n_iter_no_change
        self._device = get_torch_device(use_torch_gpu)
        self._train_print_every = train_print_every

    @abc.abstractmethod
    def forward(self, tensor_X: Tensor) -> Tensor:
        """PyTorch forward method."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _initialize_net(self) -> None:
        """Initialize the network once the data dimensions are known."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Create the loss function used for optimization."""
        raise NotImplementedError("Override me!")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create an optimizer after the model is initialized."""
        return optim.Adam(
            self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
        )

    def _fit(self, X: Array, Y: Array) -> None:
        # Initialize the network.
        self._initialize_net()
        self.to(self._device)
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self._device)
        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(self._device)
        batch_generator = single_batch_generator(tensor_X, tensor_Y)
        # Run training.
        train_pytorch_model(
            self,
            loss_fn,
            optimizer,
            batch_generator,
            device=self._device,
            print_every=self._train_print_every,
            max_train_iters=self._max_train_iters,
            dataset_size=X.shape[0],
            clip_gradients=self._clip_gradients,
            clip_value=self._clip_value,
            n_iter_no_change=self._n_iter_no_change,
        )

    def _predict(self, x: Array) -> Array:
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(self._device)
        tensor_X = tensor_x.unsqueeze(dim=0)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().cpu().numpy()
        return y  # type: ignore


class DistributionRegressor(abc.ABC):
    """ABC for classes that learn a continuous conditional sampler."""

    @abc.abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Train the model on the given data.

        X is two-dimensional, y is one-dimensional.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        """Return a sampled prediction on the given datapoint.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me!")


class MLPRegressor(PyTorchRegressor):
    """A basic multilayer perceptron regressor."""

    def __init__(
        self,
        seed: int,
        hid_sizes: list[int],
        max_train_iters: MaxTrainIters,
        clip_gradients: bool,
        clip_value: float,
        learning_rate: float,
        weight_decay: float = 0,
        use_torch_gpu: bool = False,
        train_print_every: int = 1000,
        n_iter_no_change: int = 10000000,
    ) -> None:
        super().__init__(
            seed,
            max_train_iters,
            clip_gradients,
            clip_value,
            learning_rate,
            weight_decay=weight_decay,
            n_iter_no_change=n_iter_no_change,
            use_torch_gpu=use_torch_gpu,
            train_print_every=train_print_every,
        )
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return tensor_X

    def _initialize_net(self) -> None:
        assert len(self._x_dims) == 1, "X should be two-dimensional"
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(self._x_dims[0], self._hid_sizes[0]))
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], self._y_dim))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.MSELoss()


class ImplicitMLPRegressor(PyTorchRegressor):
    """A regressor implemented via an energy function.

    For each positive (x, y) pair, a number of "negative" (x, y') pairs are
    generated. The model is then trained to distinguish positive from negative
    conditioned on x using a contrastive loss.

    The implementation idea is the following. We want to use a contrastive
    loss that looks like this:

        L = E[-log(p(y | x, {y'}))]

        p(y | x, {y'})) = exp(-f(x, y)) / [
            (exp(-f(x, y)) + sum_{y'} exp(-f(x, y')))
        ]

    where (x, y) is an example "positive" input/output from (X, Y), f is
    the energy function that we are learning in this class, and {y'} is a set
    of "negative" output examples for input x. The size of that set is
    self._num_negatives_per_input.

    One way to interpret the expression is that the numerator exp(-f(x, y))
    represents an unnormalized probability that this (x, y) belongs to
    a certain ground truth "class". Each of the exp(-f(x, y')) in the
    denominator then corresponds to an artificial incorrect "class".
    So the entire expression is just a softmax over (num_negatives + 1)
    classes.

    Inference with the "sample_once" method samples a fixed number of possible
    inputs and returns the sample that has the highest probability of
    classifying to 1, under the learned classifier.

    Inference with the "derivative_free" method follows Algorithm 1 from the
    implicit BC paper (https://arxiv.org/pdf/2109.00137.pdf). It is very
    similar to CEM.

    Inference with the "grid" method is similar to "sample_once", except that
    the samples are evenly distributed over the Y space. Note that this method
    ignores the num_samples_per_inference keyword argument and instead uses the
    grid_num_ticks_per_dim.
    """

    def __init__(
        self,
        seed: int,
        hid_sizes: list[int],
        max_train_iters: MaxTrainIters,
        clip_gradients: bool,
        clip_value: float,
        learning_rate: float,
        num_samples_per_inference: int,
        num_negative_data_per_input: int,
        temperature: float,
        inference_method: str,
        weight_decay: float = 0,
        use_torch_gpu: bool = False,
        train_print_every: int = 1000,
        derivative_free_num_iters: int | None = None,
        derivative_free_sigma_init: float | None = None,
        derivative_free_shrink_scale: float | None = None,
        grid_num_ticks_per_dim: int | None = None,
    ) -> None:
        super().__init__(
            seed,
            max_train_iters,
            clip_gradients,
            clip_value,
            learning_rate,
            weight_decay=weight_decay,
            use_torch_gpu=use_torch_gpu,
            train_print_every=train_print_every,
        )
        self._inference_method = inference_method
        self._derivative_free_num_iters = derivative_free_num_iters
        self._derivative_free_sigma_init = derivative_free_sigma_init
        self._derivative_free_shrink_scale = derivative_free_shrink_scale
        self._grid_num_ticks_per_dim = grid_num_ticks_per_dim
        self._hid_sizes = hid_sizes
        self._num_samples_per_inference = num_samples_per_inference
        self._num_negatives_per_input = num_negative_data_per_input
        self._temperature = temperature
        # Set in fit().
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        # The input here is the concatenation of the regressor's input and a
        # candidate output. A better name would be tensor_XY, but we leave it
        # as tensor_X for consistency with the parent class.
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return tensor_X.squeeze(dim=-1)

    def _initialize_net(self) -> None:
        assert len(self._x_dims) == 1, "X must be two-dimensional"
        self._linears = nn.ModuleList()
        self._linears.append(
            nn.Linear(self._x_dims[0] + self._y_dim, self._hid_sizes[0])
        )
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], 1))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:

        # See the class docstring for context.
        def _loss_fn(Y_hat: Tensor, Y: Tensor) -> Tensor:
            # The shape of Y_hat is (num_samples * (num_negatives + 1), ).
            # The shape of Y is (num_samples, (num_negatives + 1)).
            # Each row of Y is a one-hot vector with the first entry 1. We
            # could reconstruct that here, but we stick with this to conform
            # to the _train_pytorch_model API, where target outputs are always
            # passed into the loss function.
            pred = Y_hat.reshape(Y.shape)
            log_probs = F.log_softmax(pred / self._temperature, dim=-1)
            # Note: batchmean is recommended in the PyTorch documentation
            # and will become the default in a future version.
            loss = F.kl_div(log_probs, Y, reduction="batchmean")
            return loss

        return _loss_fn

    def _create_batch_generator(
        self, X: Array, Y: Array
    ) -> Iterator[tuple[Tensor, Tensor]]:
        num_samples = X.shape[0]
        num_negatives = self._num_negatives_per_input
        # Cast to torch first.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self._device)
        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(self._device)
        assert tensor_X.shape == (num_samples, *self._x_dims)
        assert tensor_Y.shape == (num_samples, self._y_dim)
        # Expand tensor_Y in preparation for concat in the loop below.
        tensor_Y = tensor_Y[:, None, :]
        assert tensor_Y.shape == (num_samples, 1, self._y_dim)
        # For each of the negative outputs, we need a corresponding input.
        # So we repeat each x value num_negatives + 1 times so that each of
        # the num_negatives outputs, and the 1 positive output, have a
        # corresponding input.
        tiled_X = tensor_X.unsqueeze(1).repeat(1, num_negatives + 1, 1)
        assert tiled_X.shape == (num_samples, num_negatives + 1, *self._x_dims)
        extended_X = tiled_X.reshape([-1, tensor_X.shape[-1]])
        assert extended_X.shape == (num_samples * (num_negatives + 1), *self._x_dims)
        while True:
            # Resample negative examples on each iteration.
            neg_Y = torch.rand(
                size=(num_samples, num_negatives, self._y_dim), dtype=tensor_Y.dtype
            )
            # Create a multiclass classification-style target vector.
            combined_Y = torch.cat([tensor_Y, neg_Y], axis=1)  # type: ignore
            combined_Y = combined_Y.reshape([-1, tensor_Y.shape[-1]])
            # Concatenate to create the final input to the network.
            XY = torch.cat([extended_X, combined_Y], axis=1)  # type: ignore
            assert XY.shape == (
                num_samples * (num_negatives + 1),
                self._x_dims[0] + self._y_dim,
            )
            # Create labels for multiclass loss. Note that the true inputs
            # are first, so the target labels are all zeros (see docstring).
            idxs = torch.zeros([num_samples], dtype=torch.int64)
            labels = F.one_hot(idxs, num_classes=num_negatives + 1).float()
            assert labels.shape == (num_samples, num_negatives + 1)
            # Note that XY is flattened and labels is not. XY is flattened
            # because we need to feed each entry through the network during
            # training. Labels is unflattened because we will want to use
            # F.kl_div in the loss function.
            yield (XY, labels)

    def _fit(self, X: Array, Y: Array) -> None:
        # Note: we need to override _fit() because we are not just training
        # a network that maps X to Y, but rather, training a network that
        # maps concatenated X and Y vectors to floats (energies).
        # Initialize the network.
        self._initialize_net()
        self.to(self._device)
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Create the batch generator, which creates negative data.
        batch_generator = self._create_batch_generator(X, Y)
        # Run training.
        train_pytorch_model(
            self,
            loss_fn,
            optimizer,
            batch_generator,
            device=self._device,
            max_train_iters=self._max_train_iters,
            dataset_size=X.shape[0],
            clip_gradients=self._clip_gradients,
            clip_value=self._clip_value,
        )

    def _predict(self, x: Array) -> Array:
        assert x.shape == self._x_dims
        if self._inference_method == "sample_once":
            return self._predict_sample_once(x)
        if self._inference_method == "derivative_free":
            return self._predict_derivative_free(x)
        if self._inference_method == "grid":
            return self._predict_grid(x)
        raise NotImplementedError(
            "Unrecognized inference method: " f"{self._inference_method}."
        )

    def _predict_sample_once(self, x: Array) -> Array:
        # This sampling-based inference method is okay in 1 dimension, but
        # won't work well with higher dimensions.
        num_samples = self._num_samples_per_inference
        sample_ys = self._rng.uniform(size=(num_samples, self._y_dim))
        # Concatenate the x and ys.
        concat_xy = np.array([np.hstack([x, y]) for y in sample_ys], dtype=np.float32)
        assert concat_xy.shape == (num_samples, self._x_dims[0] + self._y_dim)
        # Pass through network.
        scores = self(torch.from_numpy(concat_xy).to(self._device))
        # Find the highest probability sample.
        sample_idx = torch.argmax(scores)
        return sample_ys[sample_idx]  # type: ignore

    def _predict_derivative_free(self, x: Array) -> Array:
        # Reference: https://arxiv.org/pdf/2109.00137.pdf (Algorithm 1).
        # This method reportedly works well in up to 5 dimensions.
        # Since we are using torch for random sampling, and since we want
        # to ensure deterministic predictions, we need to reseed torch.
        # Also note that we need to set the seed here because we need calls
        # on the same input to deterministically return the same output,
        # both when saved models are loaded, but also when the same model
        # is called multiple times in the same process. The latter case
        # happens when an option is called by the default option model and
        # then later called at execution time.
        torch.manual_seed(self._seed)
        num_samples = self._num_samples_per_inference
        num_iters = self._derivative_free_num_iters
        sigma = self._derivative_free_sigma_init
        K = self._derivative_free_shrink_scale
        assert num_samples is not None and num_samples > 0
        assert num_iters is not None and num_iters > 0
        assert sigma is not None and sigma > 0
        assert K is not None and 0 < K < 1
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(self._device)
        repeated_x = tensor_x.repeat(num_samples, 1)
        # Initialize candidate outputs.
        Y = torch.rand(size=(num_samples, self._y_dim), dtype=tensor_x.dtype)
        for it in range(num_iters):
            # Compute candidate scores.
            concat_xy = torch.cat([repeated_x, Y], axis=1)  # type: ignore
            scores = self(concat_xy)
            if it < num_iters - 1:
                # Multinomial resampling with replacement.
                dist = Categorical(logits=scores)  # type: ignore
                indices = dist.sample((num_samples,))  # type: ignore
                Y = Y[indices]
                # Add noise.
                noise = torch.randn(Y.shape) * sigma
                Y = Y + noise
                # Recall that Y is normalized to stay within [0, 1].
                Y = torch.clip(Y, 0.0, 1.0)
                sigma = K * sigma
        # Make a final selection.
        selected_idx = torch.argmax(scores)
        return Y[selected_idx].detach().cpu().numpy()  # type: ignore

    def _predict_grid(self, x: Array) -> Array:
        assert self._grid_num_ticks_per_dim is not None
        assert self._grid_num_ticks_per_dim > 0
        dy = 1.0 / self._grid_num_ticks_per_dim
        ticks = [np.arange(0.0, 1.0, dy)] * self._y_dim
        grid = np.meshgrid(*ticks)
        candidate_ys = np.transpose(grid).reshape((-1, self._y_dim))
        num_samples = candidate_ys.shape[0]
        assert num_samples == self._grid_num_ticks_per_dim**self._y_dim
        # Concatenate the x and ys.
        concat_xy = np.array(
            [np.hstack([x, y]) for y in candidate_ys], dtype=np.float32
        )
        assert concat_xy.shape == (num_samples, self._x_dims[0] + self._y_dim)
        # Pass through network.
        scores = self(torch.from_numpy(concat_xy).to(self._device))
        # Find the highest probability sample.
        sample_idx = torch.argmax(scores)
        return candidate_ys[sample_idx]  # type: ignore


class CNNRegressor(PyTorchRegressor):
    """A basic CNN regressor operating on 2D images with multiple channels."""

    def __init__(
        self,
        seed: int,
        conv_channel_nums: list[int],
        conv_kernel_sizes: list[int],
        linear_hid_sizes: list[int],
        max_train_iters: MaxTrainIters,
        clip_gradients: bool,
        clip_value: float,
        learning_rate: float,
        weight_decay: float = 0,
        use_torch_gpu: bool = False,
        train_print_every: int = 1000,
    ) -> None:
        """Create a CNNRegressor.

        conv_channel_nums and conv_kernel_sizes define the sizes of the
        output channels and square kernels for the Conv2d layers.
        linear_hid_sizes is the same as hid_sizes for MLPRegressor.
        """
        super().__init__(
            seed,
            max_train_iters,
            clip_gradients,
            clip_value,
            learning_rate,
            weight_decay=weight_decay,
            use_torch_gpu=use_torch_gpu,
            train_print_every=train_print_every,
        )
        assert len(conv_channel_nums) == len(conv_kernel_sizes)
        self._conv_channel_nums = conv_channel_nums
        self._conv_kernel_sizes = conv_kernel_sizes
        self._linear_hid_sizes = linear_hid_sizes

        self._max_pool = nn.MaxPool2d(2, 2)
        # Set in fit().
        self._convs = nn.ModuleList()
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        for _, conv in enumerate(self._convs):
            tensor_X = self._max_pool(F.relu(conv(tensor_X)))
        tensor_X = torch.flatten(tensor_X, 1)
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return tensor_X

    def _initialize_net(self) -> None:
        self._convs = nn.ModuleList()

        # We need to calculate the size of the tensor outputted from the Conv2d
        # layers to use as the input dim for the linear layers post-flatten.
        assert len(self._x_dims) == 3, "X should be 4-dimensional (N, C, H, W)"
        c_dim, h_dim, w_dim = self._x_dims
        for i in range(len(self._conv_channel_nums)):
            kernel_size = self._conv_kernel_sizes[i]
            self._convs.append(
                nn.Conv2d(c_dim, self._conv_channel_nums[i], kernel_size)
            )
            # Calculate size after Conv2d + MaxPool2d
            c_dim = self._conv_channel_nums[i]
            h_dim = (h_dim - kernel_size + 1) // 2
            w_dim = (w_dim - kernel_size + 1) // 2

        flattened_size = c_dim * h_dim * w_dim
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(flattened_size, self._linear_hid_sizes[0]))
        for i in range(len(self._linear_hid_sizes) - 1):
            self._linears.append(
                nn.Linear(self._linear_hid_sizes[i], self._linear_hid_sizes[i + 1])
            )
        self._linears.append(nn.Linear(self._linear_hid_sizes[-1], self._y_dim))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.MSELoss()


class NeuralGaussianRegressor(PyTorchRegressor, DistributionRegressor):
    """NeuralGaussianRegressor definition."""

    def __init__(
        self,
        seed: int,
        hid_sizes: list[int],
        max_train_iters: MaxTrainIters,
        clip_gradients: bool,
        clip_value: float,
        learning_rate: float,
        weight_decay: float = 0,
        use_torch_gpu: bool = False,
        train_print_every: int = 1000,
    ) -> None:
        super().__init__(
            seed,
            max_train_iters,
            clip_gradients,
            clip_value,
            learning_rate,
            weight_decay=weight_decay,
            use_torch_gpu=use_torch_gpu,
            train_print_every=train_print_every,
        )
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        # Force pred var positive.
        # Note: use of elu here is very important. Tried several other things
        # and none worked. Use of elu recommended here:
        # https://engineering.taboola.com/predicting-probability-distributions/
        mean, variance = self._split_prediction(tensor_X)
        variance = F.elu(variance) + 1
        return torch.cat([mean, variance], dim=-1)

    def _initialize_net(self) -> None:
        # Versus MLPRegressor, the only difference here is that the output
        # size is 2 * self._y_dim, rather than self._y_dim, because we are
        # predicting both mean and diagonal variance.
        assert len(self._x_dims) == 1, "X should be two-dimensional"
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(self._x_dims[0], self._hid_sizes[0]))
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], 2 * self._y_dim))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        _nll_loss = nn.GaussianNLLLoss()

        def _loss_fn(Y_hat: Tensor, Y: Tensor) -> Tensor:
            pred_mean, pred_var = self._split_prediction(Y_hat)
            return _nll_loss(pred_mean, Y, pred_var)  # type: ignore

        return _loss_fn

    def predict_mean(self, x: Array) -> Array:
        """Return a mean prediction on the given datapoint.

        x is single-dimensional.
        """
        assert x.ndim == 1
        mean, _ = self._predict_mean_var(x)
        return mean

    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        """Return a sampled prediction on the given datapoint.

        x is single-dimensional.
        """
        assert x.ndim == 1
        mean, variance = self._predict_mean_var(x)
        y = []
        for mu, sigma_sq in zip(mean, variance):
            y_i = rng.normal(loc=mu, scale=np.sqrt(sigma_sq))
            y.append(y_i)
        return np.array(y)

    def _predict_mean_var(self, x: Array) -> tuple[Array, Array]:
        # Note: we need to use _predict(), rather than predict(), because
        # we need to apply normalization separately to the mean and variance
        # components of the prediction (see below).
        assert x.shape == self._x_dims
        # Normalize.
        norm_x = (x - self._input_shift) / self._input_scale
        norm_y = self._predict(norm_x)
        assert norm_y.shape == (2 * self._y_dim,)
        norm_mean = norm_y[: self._y_dim]
        norm_variance = norm_y[self._y_dim :]
        # Denormalize output.
        mean = (norm_mean * self._output_scale) + self._output_shift
        variance = norm_variance * (np.square(self._output_scale))
        return mean, variance

    @staticmethod
    def _split_prediction(Y: Tensor) -> tuple[Tensor, Tensor]:
        return torch.split(Y, Y.shape[-1] // 2, dim=-1)  # type: ignore


class DegenerateMLPDistributionRegressor(MLPRegressor, DistributionRegressor):
    """A model that can be used as a DistributionRegressor, but that always
    returns the same output given the same input.

    Implemented as an MLPRegressor().
    """

    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        del rng  # unused
        return self.predict(x)


class KNeighborsRegressor(_ScikitLearnRegressor):
    """K nearest neighbors from scikit-learn."""

    def _initialize_model(self, **kwargs: Any) -> BaseEstimator:
        return _SKLearnKNeighborsRegressor(**kwargs)
