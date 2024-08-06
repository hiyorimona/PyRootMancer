from src.models.model_evaluation import f1, iou
import unittest
import tensorflow as tf
import numpy as np
import pytest
import tensorflow.keras.backend as K
from unittest.mock import patch, MagicMock, call
import importlib

tf.experimental.numpy.experimental_enable_numpy_behavior()

def recall_m(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (Positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_Positives + K.epsilon())
    return precision

@pytest.mark.parametrize("y_true, y_pred, expected_recall", [
    (np.array([[[[1]]]]), np.array([[[[1]]]]), 1.0),
    (np.array([[[[0]]]]), np.array([[[[1]]]]), 0.0),
    (np.array([[[[1, 0], [0, 1]]]]), np.array([[[[0, 1], [1, 0]]]]), 0.0),
    (np.array([[[[1, 1], [1, 1]]]]), np.array([[[[1, 1], [1, 1]]]]), 1.0),
    (np.array([[[[1, 1], [0, 0]]]]), np.array([[[[0, 0], [1, 1]]]]), 0.0)
])
def test_recall_m(y_true, y_pred, expected_recall):
    y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    result = recall_m(y_true_tf, y_pred_tf).numpy()
    assert np.isclose(result, expected_recall, atol=1e-7)

@pytest.mark.parametrize("y_true, y_pred, expected_precision", [
    (np.array([[[[1]]]]), np.array([[[[1]]]]), 1.0),
    (np.array([[[[0]]]]), np.array([[[[1]]]]), 0.0),
    (np.array([[[[1, 0], [0, 1]]]]), np.array([[[[0, 1], [1, 0]]]]), 0.0),
    (np.array([[[[1, 1], [1, 1]]]]), np.array([[[[1, 1], [1, 1]]]]), 1.0),
    (np.array([[[[1, 1], [0, 0]]]]), np.array([[[[0, 0], [1, 1]]]]), 0.0)
])
def test_precision_m(y_true, y_pred, expected_precision):
    y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    result = precision_m(y_true_tf, y_pred_tf).numpy()
    assert np.isclose(result, expected_precision, atol=1e-7)

@pytest.mark.parametrize("y_true, y_pred, expected_f1", [
    (np.array([[[[1]]]]), np.array([[[[1]]]]), 1.0),
    (np.array([[[[0]]]]), np.array([[[[1]]]]), 0.0),
    (np.array([[[[1, 0], [0, 1]]]]), np.array([[[[0, 1], [1, 0]]]]), 0.0),
    (np.array([[[[1, 1], [1, 1]]]]), np.array([[[[1, 1], [1, 1]]]]), 1.0),
    (np.array([[[[1, 1], [0, 0]]]]), np.array([[[[0, 0], [1, 1]]]]), 0.0)
])
def test_f1(y_true, y_pred, expected_f1):
    y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    result = f1(y_true_tf, y_pred_tf).numpy()
    assert np.isclose(result, expected_f1, atol=1e-7)

@pytest.mark.parametrize("y_true, y_pred, expected_iou", [
    (np.array([[[[1]]]]), np.array([[[[1]]]]), 1.0),
    (np.array([[[[0]]]]), np.array([[[[1]]]]), 0.0),
    (np.array([[[[1, 0], [0, 1]]]]), np.array([[[[0, 1], [1, 0]]]]), 0.0),
    (np.array([[[[1, 1], [1, 1]]]]), np.array([[[[1, 1], [1, 1]]]]), 1.0),
    (np.array([[[[1, 1], [0, 0]]]]), np.array([[[[0, 0], [1, 1]]]]), 0.0)
])
def test_iou(y_true, y_pred, expected_iou):
    y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    result = iou(y_true_tf, y_pred_tf).numpy()
    assert np.isclose(result, expected_iou, atol=1e-7)

if __name__ == "__main__":
    pytest.main()