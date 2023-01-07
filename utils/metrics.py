import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


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


def get_acc_from_csv(fname):
    df = pd.read_csv(fname)
    y_true = df["label"].values
    y_pred = df[["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"]].values
    return get_acc(y_true, y_pred)


def save_confusion_matrix(y_true, y_pred, fname):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    labels = ["0", "1", "2", "3", "4"]
    cm = pd.DataFrame(data=cm, index=labels, columns=labels)
    cm_normalized = cm.div(cm.sum(axis=1), axis=0)

    sns.heatmap(cm_normalized, square=True, cbar=True, annot=True, cmap="Blues")
    plt.yticks(rotation=0)
    plt.xlabel("Pred", fontsize=13, rotation=0)
    plt.ylabel("GT", fontsize=13)
    ax.set_ylim(len(cm_normalized), 0)
    plt.savefig(fname)


if __name__ == "__main__":
    fname = "oof_ch7_230102_aug.csv"
    df = pd.read_csv(f"data/output/oofs/{fname}")
    y_true = df["label"].values
    y_pred = df[["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"]].values
    print(y_true.shape, y_pred.shape)
    print(get_acc(y_true, y_pred))
