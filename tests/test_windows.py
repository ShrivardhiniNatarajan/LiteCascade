import numpy as np
import pandas as pd

from litecascade.data.windows import create_windows


def test_create_windows():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "label_binary": [0, 0, 1, 1, 1, 0, 0, 0, 1],
            "timestamp": pd.date_range("2026-01-01", periods=9),
        }
    )

    t_size = 2
    stride = 1
    x_windows, y = create_windows(df, ["feature_1"], T=t_size, stride=stride)

    # A has 3 records -> 2 windows
    # B has 2 records -> 1 window
    # C has 4 records -> 3 windows
    # Total windows = 6
    assert len(x_windows) == 6
    assert len(y) == 6

    # Check X shape: (6, 2, 1)
    assert x_windows.shape == (6, t_size, 1)

    # First window of A: records [1, 2], label of last record is 0
    assert np.array_equal(x_windows[0, :, 0], [1, 2])
    assert y[0] == 0

    # Second window of A: records [2, 3], label is 1
    assert np.array_equal(x_windows[1, :, 0], [2, 3])
    assert y[1] == 1

    # Window of B: records [4, 5], label is 1
    assert np.array_equal(x_windows[2, :, 0], [4, 5])
    assert y[2] == 1

    # Ensure window never crosses boundary by asserting shape exactly matches
