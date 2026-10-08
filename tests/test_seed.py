import random

import numpy as np
import torch

from litecascade.utils.seed import set_seed


def test_set_seed_reproducibility():
    """Test that setting a seed yields reproducible random values."""
    set_seed(42)

    val_py_1 = random.randint(0, 1000)
    val_np_1 = np.random.rand()
    val_torch_1 = torch.rand(1).item()

    # Reset seed and re-generate
    set_seed(42)

    val_py_2 = random.randint(0, 1000)
    val_np_2 = np.random.rand()
    val_torch_2 = torch.rand(1).item()

    assert val_py_1 == val_py_2
    assert np.isclose(val_np_1, val_np_2)
    assert np.isclose(val_torch_1, val_torch_2)


def test_set_seed_different_seeds():
    """Test that different seeds yield different values."""
    set_seed(42)
    val_np_1 = np.random.rand()

    set_seed(43)
    val_np_2 = np.random.rand()

    assert not np.isclose(val_np_1, val_np_2)
