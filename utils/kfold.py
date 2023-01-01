from sklearn.model_selection import StratifiedGroupKFold

from parameters.variables import LABEL2ID


def create_folds(train_df, n_splits=5):
    if "target" in train_df.columns:
        train_df["target"] = train_df["target"].map(lambda x: LABEL2ID[x])
        train_df = train_df.rename(columns={"target": "label"})

    X = train_df.copy()
    y = train_df.label
    groups = train_df.id
    cv = StratifiedGroupKFold(n_splits=n_splits)
    train_df["fold"] = 0
    for fold_idx, (train_idxs, test_idxs) in enumerate(cv.split(X, y, groups)):
        train_df.loc[test_idxs, "fold"] = fold_idx
    return train_df
