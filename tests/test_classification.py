"""Tests for classification.py."""

import logging
import time
from unittest.mock import patch

import numpy as np
import pytest

from ml_helpers.classification import (
    BinaryClassifierEnsemble,
    KNeighborsClassifier,
    MLPBinaryClassifier,
)


def test_mlp_classifier():
    """Tests for MLPBinaryClassifier()."""
    input_size = 3
    num_class_samples = 5
    X = np.concatenate(
        [
            np.zeros((num_class_samples, input_size)),
            np.ones((num_class_samples, input_size)),
        ]
    )
    y = np.concatenate([np.zeros((num_class_samples)), np.ones((num_class_samples))])
    model = MLPBinaryClassifier(
        seed=123,
        balance_data=True,
        max_train_iters=100,
        learning_rate=1e-3,
        n_iter_no_change=1000000,
        hid_sizes=[32, 32],
        n_reinitialize_tries=1,
        weight_init="default",
    )
    model.fit(X, y)
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    assert model.predict_proba(np.zeros(input_size)) < 0.5
    prediction = model.classify(np.ones(input_size))
    assert prediction
    assert model.predict_proba(np.ones(input_size)) > 0.5
    # Test for early stopping
    model = MLPBinaryClassifier(
        seed=123,
        balance_data=True,
        max_train_iters=100000,
        learning_rate=1e-2,
        n_iter_no_change=-1,
        hid_sizes=[32, 32],
        n_reinitialize_tries=1,
        weight_init="default",
        train_print_every=1,
    )
    with patch.object(logging, "info", return_value=None) as mock_logging_info:
        model.fit(X, y)
    assert mock_logging_info.call_count < 5
    # Test with no positive examples.
    num_class_samples = 1000
    X = np.concatenate(
        [
            np.zeros((num_class_samples, input_size)),
            np.ones((num_class_samples, input_size)),
        ]
    )
    y = np.zeros(len(X))
    model = MLPBinaryClassifier(
        seed=123,
        balance_data=True,
        max_train_iters=100000,
        learning_rate=1e-3,
        n_iter_no_change=100000,
        hid_sizes=[32, 32],
        n_reinitialize_tries=1,
        weight_init="default",
    )
    start_time = time.perf_counter()
    model.fit(X, y)
    assert time.perf_counter() - start_time < 1, "Fitting was not instantaneous"
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    prediction = model.classify(np.ones(input_size))
    assert not prediction
    proba = model.predict_proba(np.zeros(input_size))
    assert abs(proba - 0.0) < 1e-6
    # Test with no negative examples.
    y = np.ones(len(X))
    model = MLPBinaryClassifier(
        seed=123,
        balance_data=True,
        max_train_iters=100000,
        learning_rate=1e-3,
        n_iter_no_change=100000,
        hid_sizes=[32, 32],
        n_reinitialize_tries=1,
        weight_init="default",
    )
    start_time = time.perf_counter()
    model.fit(X, y)
    assert time.perf_counter() - start_time < 1, "Fitting was not instantaneous"
    prediction = model.classify(np.zeros(input_size))
    assert prediction
    prediction = model.classify(np.ones(input_size))
    assert prediction
    proba = model.predict_proba(np.zeros(input_size))
    assert abs(proba - 1.0) < 1e-6
    # Test with non-default weight initialization.
    X = np.concatenate(
        [
            np.zeros((num_class_samples, input_size)),
            np.ones((num_class_samples, input_size)),
        ]
    )
    y = np.concatenate([np.zeros((num_class_samples)), np.ones((num_class_samples))])
    model = MLPBinaryClassifier(
        seed=123,
        balance_data=True,
        max_train_iters=100,
        learning_rate=1e-3,
        n_iter_no_change=100000,
        hid_sizes=[32, 32],
        n_reinitialize_tries=1,
        weight_init="normal",
    )
    model.fit(X, y)
    # Test with invalid weight initialization.
    model = MLPBinaryClassifier(
        seed=123,
        balance_data=True,
        max_train_iters=100000,
        learning_rate=1e-3,
        n_iter_no_change=100000,
        hid_sizes=[32, 32],
        n_reinitialize_tries=1,
        weight_init="foo",
    )
    with pytest.raises(NotImplementedError):
        model.fit(X, y)
    # Test for reinitialization failure.
    model = MLPBinaryClassifier(
        seed=123,
        balance_data=True,
        max_train_iters=100000,
        learning_rate=1e-3,
        n_iter_no_change=100000,
        hid_sizes=[32, 32],
        n_reinitialize_tries=0,
        weight_init="default",
    )
    with pytest.raises(RuntimeError):
        model.fit(X, y)


def test_binary_classifier_ensemble():
    """Tests for BinaryClassifierEnsemble()."""
    input_size = 3
    num_class_samples = 5
    X = np.concatenate(
        [
            np.zeros((num_class_samples, input_size)),
            np.ones((num_class_samples, input_size)),
        ]
    )
    y = np.concatenate([np.zeros((num_class_samples)), np.ones((num_class_samples))])
    model = BinaryClassifierEnsemble(
        seed=123,
        ensemble_size=3,
        member_cls=MLPBinaryClassifier,
        balance_data=True,
        max_train_iters=100,
        learning_rate=1e-3,
        n_iter_no_change=1000000,
        hid_sizes=[32, 32],
        n_reinitialize_tries=1,
        weight_init="default",
    )
    model.fit(X, y)
    with pytest.raises(Exception) as e:
        model.predict_proba(np.zeros(input_size))
    assert "Can't call predict_proba()" in str(e)
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    probas = model.predict_member_probas(np.zeros(input_size))
    assert all(p < 0.5 for p in probas)
    assert len(probas) == 3
    assert probas[0] != probas[1]  # there should be some variation
    prediction = model.classify(np.ones(input_size))
    assert prediction
    probas = model.predict_member_probas(np.ones(input_size))
    assert all(p > 0.5 for p in probas)
    assert len(probas) == 3
    # Test the KNN classifier with n_neighbors = num_class_samples.
    # Since there are num_class_samples data points of each class,
    # the probas should be all 0's or all 1's.
    model = BinaryClassifierEnsemble(
        seed=123,
        ensemble_size=3,
        member_cls=KNeighborsClassifier,
        n_neighbors=num_class_samples,
    )
    model.fit(X, y)
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    probas = model.predict_member_probas(np.zeros(input_size))
    assert all(p == 0.0 for p in probas)
    assert len(probas) == 3
    prediction = model.classify(np.ones(input_size))
    assert prediction
    probas = model.predict_member_probas(np.ones(input_size))
    assert all(p == 1.0 for p in probas)
    assert len(probas) == 3
    # Test the KNN classifier with n_neighbors = 2 * num_class_samples.
    # Since there are num_class_samples data points of each class,
    # the probas should be all 0.5's.
    model = BinaryClassifierEnsemble(
        seed=123,
        ensemble_size=3,
        member_cls=KNeighborsClassifier,
        n_neighbors=(2 * num_class_samples),
    )
    model.fit(X, y)
    probas = model.predict_member_probas(np.zeros(input_size))
    assert all(p == 0.5 for p in probas)
    assert len(probas) == 3
    probas = model.predict_member_probas(np.ones(input_size))
    assert all(p == 0.5 for p in probas)
    assert len(probas) == 3


def test_k_neighbors_classifier():
    """Tests for KNeighborsClassifier()."""
    input_size = 3
    num_samples = 5
    model = KNeighborsClassifier(seed=123, n_neighbors=1)
    rng = np.random.default_rng(123)
    X = rng.normal(size=(num_samples, input_size))
    Y = rng.choice(2, size=(num_samples,))
    model.fit(X, Y)
    x = X[0]
    predicted_y = model.classify(x)
    expected_y = Y[0]
    assert isinstance(predicted_y, bool)
    assert predicted_y == expected_y
    assert model.predict_proba(x) == expected_y
    # Test with no negative examples.
    Y = np.ones_like(Y)
    model = KNeighborsClassifier(seed=123, n_neighbors=1)
    model.fit(X, Y)
    x = X[0]
    assert model.classify(x) == 1
    assert model.predict_proba(x) == 1
