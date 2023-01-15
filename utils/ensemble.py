import pandas as pd


def ensemble_predictions(pred_file_paths):
    pred = None
    accepted_preds = [
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3/4",
        "Sleep stage W",
        "Sleep stage R",
    ]
    for pred_file_path in pred_file_paths:
        df = pd.read_csv(pred_file_path)
        if pred is None:
            pred = df
        else:
            pred = pd.concat([pred, df], axis=1)
    # predの中で最も多く出現する値を取得する
    pred = pred.mode(axis=1)[0]
    # predの中にaccepted_preds以外が含まれている場合エラー
    if not all([p in accepted_preds for p in pred]):
        raise ValueError("Invalid prediction")
    return pred
