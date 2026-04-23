"""
metrics.py
----------
Các độ đo đánh giá kết quả phân cụm (clustering evaluation metrics).

Bao gồm:
    - Clustering Accuracy (ACC) với Hungarian matching
    - Normalized Mutual Information (NMI)
    - Adjusted Rand Index (ARI)
"""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính Clustering Accuracy (ACC) sử dụng Hungarian algorithm để tìm
    ánh xạ tốt nhất giữa nhãn dự đoán và nhãn thật.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Nhãn thật (ground truth labels).
    y_pred : np.ndarray, shape (n,)
        Nhãn dự đoán từ thuật toán phân cụm.

    Returns
    -------
    float
        Giá trị ACC trong khoảng [0, 1].
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    assert y_true.shape == y_pred.shape, "y_true và y_pred phải cùng kích thước"

    n_classes = max(y_true.max(), y_pred.max()) + 1

    # Xây dựng ma trận confusion (cost matrix)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    # Hungarian matching tối đa hóa tổng
    row_ind, col_ind = linear_sum_assignment(-confusion)
    acc = confusion[row_ind, col_ind].sum() / y_true.shape[0]
    return float(acc)


def normalized_mutual_information(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính Normalized Mutual Information (NMI).

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Nhãn thật.
    y_pred : np.ndarray, shape (n,)
        Nhãn dự đoán.

    Returns
    -------
    float
        Giá trị NMI trong khoảng [0, 1].
    """
    return float(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))


def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính Adjusted Rand Index (ARI).

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Nhãn thật.
    y_pred : np.ndarray, shape (n,)
        Nhãn dự đoán.

    Returns
    -------
    float
        Giá trị ARI trong khoảng [-1, 1], càng cao càng tốt.
    """
    return float(adjusted_rand_score(y_true, y_pred))


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính toàn bộ các độ đo phân cụm: ACC, NMI, ARI.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật.
    y_pred : np.ndarray
        Nhãn dự đoán.

    Returns
    -------
    dict
        Dictionary chứa {'ACC': float, 'NMI': float, 'ARI': float}.
    """
    return {
        'ACC': clustering_accuracy(y_true, y_pred),
        'NMI': normalized_mutual_information(y_true, y_pred),
        'ARI': adjusted_rand_index(y_true, y_pred),
    }
