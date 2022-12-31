from sklearn.metrics import roc_auc_score


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
