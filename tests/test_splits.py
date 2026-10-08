import pandas as pd

from litecascade.data.splits import create_group_splits, create_lofo_split


def _make_dummy_df():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "label_family": [
                "benign",
                "benign",
                "mirai",
                "mirai",
                "benign",
                "benign",
                "tsunami",
                "tsunami",
            ],
            "feature": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


def test_create_group_splits():
    df = _make_dummy_df()
    # Use ratios that give at least 1 group to each
    ratios = {"train": 0.5, "test": 0.5}
    splits = create_group_splits(df, "dummy", ratios)

    train_idx = set(splits["train"])
    test_idx = set(splits["test"])

    # Assert disjoint indices
    assert train_idx.isdisjoint(test_idx)

    # Assert zero group overlap
    train_groups = set(df.loc[list(train_idx), "group"])
    test_groups = set(df.loc[list(test_idx), "group"])
    assert train_groups.isdisjoint(test_groups)


def test_create_lofo_split():
    df = _make_dummy_df()
    splits = create_lofo_split(df, "mirai", "dummy")

    train_idx = splits["train"]
    test_idx = splits["test"]

    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    # Assert test family absent from train
    assert "mirai" not in train_df["label_family"].values
    # Assert test contains test family
    assert "mirai" in test_df["label_family"].values

    # Assert zero group overlap
    assert set(train_df["group"]).isdisjoint(set(test_df["group"]))
