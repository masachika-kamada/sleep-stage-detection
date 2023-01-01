import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def AUC(y_true, y_pred, onehot=False):
    if y_pred.shape[1] == 2:
        auc = roc_auc_score(y_true, y_pred[:, 1])
    else:
        if onehot:
            auc = 0
            n_col = y_pred.shape[1]
            # print(n_col, y_true[:, i].shape)
            for i in range(n_col):
                auc += roc_auc_score(y_true[:, i], y_pred[:, i]) / n_col
        else:
            auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
    return auc


def get_acc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_true, y_pred)
    return acc


if __name__ == "__main__":
    fname = "oof_ch7_230101_fmore.csv"
    df = pd.read_csv(f"data/output/{fname}")
    y_true = df["label"].values
    y_pred = df[["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"]].values
    print(y_true.shape, y_pred.shape)
    print(get_acc(y_true, y_pred))
