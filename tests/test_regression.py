"""Tests for classification.py."""

import numpy as np
import pytest

from ml_helpers.regression import (
    CNNRegressor,
    DegenerateMLPDistributionRegressor,
    ImplicitMLPRegressor,
    KNeighborsRegressor,
    MLPRegressor,
    NeuralGaussianRegressor,
)


def test_basic_mlp_regressor():
    """Tests for MLPRegressor()."""
    input_size = 3
    output_size = 2
    num_samples = 5
    model = MLPRegressor(
        seed=123,
        hid_sizes=[32, 32],
        max_train_iters=100,
        n_iter_no_change=1000,
        clip_gradients=True,
        clip_value=5,
        learning_rate=1e-3,
    )
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = np.zeros(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-2)
    # Test with nonzero outputs.
    Y = 75 * np.ones((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = 75 * np.ones(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-2)


def test_implicit_mlp_regressor():
    """Tests for ImplicitMLPRegressor()."""
    input_size = 3
    output_size = 1
    num_samples = 5
    model = ImplicitMLPRegressor(
        seed=123,
        hid_sizes=[32, 32],
        max_train_iters=100,
        clip_gradients=False,
        clip_value=5,
        learning_rate=1e-3,
        num_samples_per_inference=100,
        num_negative_data_per_input=5,
        temperature=1.0,
        inference_method="sample_once",
        derivative_free_num_iters=3,
        derivative_free_sigma_init=0.33,
        derivative_free_shrink_scale=0.5,
        grid_num_ticks_per_dim=100,
    )
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = np.zeros(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-1)
    # Test with nonzero outputs.
    Y = 75 * np.ones((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = 75 * np.ones(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-1)
    # Test other inference methods. Protected access is to avoid retraining.
    model._inference_method = "derivative_free"  # pylint: disable=protected-access
    predicted_y = model.predict(x)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-1)
    model._inference_method = "grid"  # pylint: disable=protected-access
    predicted_y = model.predict(x)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-1)
    model._inference_method = (  # pylint: disable=protected-access
        "not a real inference method"
    )
    with pytest.raises(NotImplementedError):
        model.predict(x)


def test_basic_cnn_regressor():
    """Tests for CNNRegressor()."""
    input_size = (3, 9, 6)
    output_size = 2
    num_samples = 5
    model = CNNRegressor(
        seed=123,
        conv_channel_nums=[1, 1],
        conv_kernel_sizes=[3, 1],
        linear_hid_sizes=[32, 32],
        max_train_iters=100,
        clip_gradients=True,
        clip_value=5,
        learning_rate=1e-3,
    )
    X = np.ones((num_samples, *input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = np.zeros(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-2)
    # Test with nonzero outputs.
    Y = 75 * np.ones((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = 75 * np.ones(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-2)


def test_neural_gaussian_regressor():
    """Tests for NeuralGaussianRegressor()."""
    input_size = 3
    output_size = 2
    num_samples = 5
    model = NeuralGaussianRegressor(
        seed=123,
        hid_sizes=[32, 32],
        max_train_iters=100,
        clip_gradients=False,
        clip_value=5,
        learning_rate=1e-3,
    )
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    mean = model.predict_mean(x)
    expected_y = np.zeros(output_size)
    assert mean.shape == expected_y.shape
    assert np.allclose(mean, expected_y, atol=1e-2)
    rng = np.random.default_rng(123)
    sample = model.predict_sample(x, rng)
    assert sample.shape == expected_y.shape


def test_degenerate_mlp_distribution_regressor():
    """Tests for DegenerateMLPDistributionRegressor."""
    input_size = 3
    output_size = 2
    num_samples = 5
    model = DegenerateMLPDistributionRegressor(
        seed=123,
        hid_sizes=[32, 32],
        max_train_iters=100,
        clip_gradients=True,
        clip_value=5,
        learning_rate=1e-3,
    )
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    mean = model.predict(x)
    expected_y = np.zeros(output_size)
    assert mean.shape == expected_y.shape
    assert np.allclose(mean, expected_y, atol=1e-2)
    rng = np.random.default_rng(123)
    sample = model.predict_sample(x, rng)
    assert sample.shape == expected_y.shape
    assert np.allclose(sample, expected_y, atol=1e-2)
    assert np.allclose(sample, mean, atol=1e-6)


def test_k_neighbors_regressor():
    """Tests for KNeighborsRegressor()."""
    input_size = 3
    output_size = 2
    num_samples = 5
    model = KNeighborsRegressor(seed=123, n_neighbors=1)
    rng = np.random.default_rng(123)
    X = rng.normal(size=(num_samples, input_size))
    Y = rng.normal(size=(num_samples, output_size))
    model.fit(X, Y)
    x = X[0]
    predicted_y = model.predict(x)
    expected_y = Y[0]
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-7)
