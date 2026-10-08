import numpy as np

from litecascade.eval.metrics import bootstrap_metrics, compute_detection_metrics


def test_compute_detection_metrics():
    # Known confusion matrix:
    # TN=2 (pred=0, true=0), FP=1 (pred=1, true=0)
    # FN=1 (pred=0, true=1), TP=3 (pred=1, true=1)
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1])

    metrics = compute_detection_metrics(y_true, y_pred)

    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 3

    # Accuracy: 5/7
    assert np.isclose(metrics["accuracy"], 5 / 7)

    # Precision: TP / (TP + FP) = 3 / 4 = 0.75
    assert np.isclose(metrics["precision"], 0.75)

    # Malicious Recall: TP / (TP + FN) = 3 / 4 = 0.75
    assert np.isclose(metrics["recall_malicious"], 0.75)

    # FPR: FP / (FP + TN) = 1 / 3
    assert np.isclose(metrics["fpr"], 1 / 3)


def test_bootstrap_metrics():
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1] * 10)
    y_pred = np.array([0, 0, 1, 1] * 10)  # perfect

    ci = bootstrap_metrics(y_true, y_pred, n_bootstraps=10)

    assert ci["f1_macro_95ci"][0] == 1.0
    assert ci["f1_macro_95ci"][1] == 1.0
