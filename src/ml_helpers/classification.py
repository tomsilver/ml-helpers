"""Models for classification."""

import abc
from typing import Any, Callable, Type

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier as _SKLearnKNeighborsClassifier
from torch import Tensor, nn, optim

from ml_helpers.structs import Array, MaxTrainIters
from ml_helpers.utils import (
    balance_binary_classification_data,
    get_torch_device,
    normalize_data,
    single_batch_generator,
    train_pytorch_model,
)


class BinaryClassifier(abc.ABC):
    """ABC for binary classifier classes."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Train the classifier on the given data.

        X is two-dimensional, y is one-dimensional.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def classify(self, x: Array) -> bool:
        """Return a predicted class for the given datapoint.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def predict_proba(self, x: Array) -> float:
        """Get the predicted probability that the input classifies to 1.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me!")


class _ScikitLearnBinaryClassifier(BinaryClassifier):
    """A regressor that lightly wraps a scikit-learn classification model."""

    def __init__(self, seed: int, **kwargs: Any) -> None:
        super().__init__(seed)
        self._model = self._initialize_model(**kwargs)

    @abc.abstractmethod
    def _initialize_model(self, **kwargs: Any) -> BaseEstimator:
        raise NotImplementedError("Override me!")

    def fit(self, X: Array, y: Array) -> None:
        return self._model.fit(X, y)  # type: ignore

    def classify(self, x: Array) -> bool:
        class_prediction = self._model.predict([x])[0]
        assert class_prediction in [0, 1]
        return bool(class_prediction)

    def predict_proba(self, x: Array) -> float:
        probs = self._model.predict_proba([x])[0]
        # Special case: only one class.
        if probs.shape == (1,):
            return float(self.classify(x))
        assert probs.shape == (2,)  # [P(x is class 0), P(x is class 1)]
        return probs[1]  # type: ignore


class _NormalizingBinaryClassifier(BinaryClassifier):
    """A binary classifier that normalizes the data.

    Also infers the dimensionality of the inputs and outputs from fit().

    Also implements data balancing (optionally) and single-class
    prediction.
    """

    def __init__(self, seed: int, balance_data: bool) -> None:
        super().__init__(seed)
        self._balance_data = balance_data
        # Set in fit().
        self._x_dims: tuple[int, ...] = tuple()
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._do_single_class_prediction = False
        self._predicted_single_class = False

    def fit(self, X: Array, y: Array) -> None:
        """Train the classifier on the given data.

        X is two-dimensional, y is one-dimensional.
        """
        num_data = X.shape[0]
        self._x_dims = tuple(X.shape[1:])
        assert y.shape == (num_data,)
        print(
            f"Training {self.__class__.__name__} on {num_data} "
            f"datapoints ({sum(y)} positive)"
        )
        # If there is only one class in the data, then there's no point in
        # learning, since any predictions other than that one class could
        # only be generalization issues.
        if np.all(y == 0):
            self._do_single_class_prediction = True
            self._predicted_single_class = False
            return
        if np.all(y == 1):
            self._do_single_class_prediction = True
            self._predicted_single_class = True
            return
        # Balance the classes.
        if self._balance_data and len(y) // 2 > sum(y):
            old_len = len(y)
            X, y = balance_binary_classification_data(X, y, self._rng)
            print(f"Reduced dataset size from {old_len} to {len(y)}")
        X, self._input_shift, self._input_scale = normalize_data(X)
        self._fit(X, y)

    def classify(self, x: Array) -> bool:
        """Return a predicted class for the given datapoint.

        x is single-dimensional.
        """
        assert len(self._x_dims), "Fit must be called before classify."
        assert x.shape == self._x_dims
        if self._do_single_class_prediction:
            return self._predicted_single_class
        # Normalize.
        x = (x - self._input_shift) / self._input_scale
        # Make prediction.
        return self._classify(x)

    @abc.abstractmethod
    def _fit(self, X: Array, y: Array) -> None:
        """Train the classifier on normalized data."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _classify(self, x: Array) -> bool:
        """Return a predicted class for the normalized input."""
        raise NotImplementedError("Override me!")


class PyTorchBinaryClassifier(_NormalizingBinaryClassifier, nn.Module):
    """ABC for PyTorch binary classification models."""

    def __init__(
        self,
        seed: int,
        balance_data: bool,
        max_train_iters: MaxTrainIters,
        learning_rate: float,
        n_iter_no_change: int,
        n_reinitialize_tries: int,
        weight_init: str,
        weight_decay: float = 0,
        use_torch_gpu: bool = False,
        train_print_every: int = 1000,
    ) -> None:
        torch.manual_seed(seed)
        _NormalizingBinaryClassifier.__init__(self, seed, balance_data)
        nn.Module.__init__(self)  # type: ignore
        self._max_train_iters = max_train_iters
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._n_iter_no_change = n_iter_no_change
        self._n_reinitialize_tries = n_reinitialize_tries
        self._weight_init = weight_init
        self._device = get_torch_device(use_torch_gpu)
        self._train_print_every = train_print_every

    @abc.abstractmethod
    def forward(self, tensor_X: Tensor) -> Tensor:
        """PyTorch forward method."""
        raise NotImplementedError("Override me!")

    def predict_proba(self, x: Array) -> float:
        """Get the predicted probability that the input classifies to 1.

        The input is NOT normalized.
        """
        if self._do_single_class_prediction:
            return float(self._predicted_single_class)
        norm_x = (x - self._input_shift) / self._input_scale
        return self._forward_single_input_np(norm_x)

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

    def _reset_weights(self) -> None:
        """(Re-)initialize the network weights."""
        self.apply(lambda m: self._weight_reset(m, self._weight_init))

    def _weight_reset(self, m: torch.nn.Module, weight_init: str) -> None:
        if isinstance(m, nn.Linear):
            if weight_init == "default":
                m.reset_parameters()
            elif weight_init == "normal":
                torch.nn.init.normal_(m.weight)
            else:
                raise NotImplementedError(
                    f"{weight_init} weight initialization unknown"
                )
        else:
            # To make sure all the weights are being reset
            assert m is self or isinstance(m, nn.ModuleList)

    def _fit(self, X: Array, y: Array) -> None:
        # Initialize the network.
        self._initialize_net()
        self.to(self._device)
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self._device)
        tensor_y = torch.from_numpy(np.array(y, dtype=np.float32)).to(self._device)
        batch_generator = single_batch_generator(tensor_X, tensor_y)
        # Run training.
        for _ in range(self._n_reinitialize_tries):
            # (Re-)initialize weights.
            self._reset_weights()
            # Create the optimizer.
            optimizer = self._create_optimizer()
            # Run training.
            best_loss = train_pytorch_model(
                self,
                loss_fn,
                optimizer,
                batch_generator,
                device=self._device,
                print_every=self._train_print_every,
                max_train_iters=self._max_train_iters,
                dataset_size=X.shape[0],
                n_iter_no_change=self._n_iter_no_change,
            )
            # Weights may not have converged during training.
            if best_loss < 1:
                break  # success!
        else:
            raise RuntimeError(
                f"Failed to converge within " f"{self._n_reinitialize_tries} tries"
            )

    def _forward_single_input_np(self, x: Array) -> float:
        """Helper for _classify() and predict_proba()."""
        assert x.shape == self._x_dims
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(self._device)
        tensor_X = tensor_x.unsqueeze(dim=0)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().cpu().numpy()
        proba = y.item()
        assert 0 <= proba <= 1
        return proba  # type: ignore

    def _classify(self, x: Array) -> bool:
        return self._forward_single_input_np(x) > 0.5


class MLPBinaryClassifier(PyTorchBinaryClassifier):
    """MLPBinaryClassifier definition."""

    def __init__(
        self,
        seed: int,
        balance_data: bool,
        max_train_iters: MaxTrainIters,
        learning_rate: float,
        n_iter_no_change: int,
        hid_sizes: list[int],
        n_reinitialize_tries: int,
        weight_init: str,
        weight_decay: float = 0,
        use_torch_gpu: bool = False,
        train_print_every: int = 1000,
    ) -> None:
        super().__init__(
            seed,
            balance_data,
            max_train_iters,
            learning_rate,
            n_iter_no_change,
            n_reinitialize_tries,
            weight_init,
            weight_decay=weight_decay,
            use_torch_gpu=use_torch_gpu,
            train_print_every=train_print_every,
        )
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def _initialize_net(self) -> None:
        assert len(self._x_dims) == 1, "X should be two-dimensional"
        self._linears.append(nn.Linear(self._x_dims[0], self._hid_sizes[0]))
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], 1))
        self._reset_weights()

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.BCELoss()

    def forward(self, tensor_X: Tensor) -> Tensor:
        assert not self._do_single_class_prediction
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return torch.sigmoid(tensor_X.squeeze(dim=-1))


class KNeighborsClassifier(_ScikitLearnBinaryClassifier):
    """K nearest neighbors from scikit-learn."""

    def _initialize_model(self, **kwargs: Any) -> BaseEstimator:
        return _SKLearnKNeighborsClassifier(**kwargs)


class BinaryClassifierEnsemble(BinaryClassifier):
    """BinaryClassifierEnsemble definition."""

    def __init__(
        self,
        seed: int,
        ensemble_size: int,
        member_cls: Type[BinaryClassifier],
        **kwargs: Any,
    ) -> None:
        super().__init__(seed)
        self._members = [member_cls(seed + i, **kwargs) for i in range(ensemble_size)]

    def fit(self, X: Array, y: Array) -> None:
        for i, member in enumerate(self._members):
            print(f"Fitting member {i} of ensemble...")
            member.fit(X, y)

    def classify(self, x: Array) -> bool:
        avg = np.mean(self.predict_member_probas(x))
        classification = bool(avg > 0.5)
        return classification

    def predict_proba(self, x: Array) -> float:
        raise Exception(
            "Can't call predict_proba() on an ensemble. Use "
            "predict_member_probas() instead."
        )

    def predict_member_probas(self, x: Array) -> Array:
        """Return class probabilities predicted by each member."""
        return np.array([m.predict_proba(x) for m in self._members])
