import numpy as np
import pandas as pd
from tqdm import tqdm

from parameters.data_path import DIR_ARRAYS, DIR_PROCESSED
from utils.kfold import create_folds


def main():
    df = pd.read_csv(f"{DIR_PROCESSED}/train_df_fold.csv")
    df = df.drop(columns=["fold"])
    df["epoch"] = df["epoch"].astype(int)
    df["meas_time"] = pd.to_datetime(df["meas_time"])

    data_cross = []

    # idを抽出
    ids = df["id"].unique()
    for label in [1, 3]:
        for i in tqdm(ids):
            tmp = df[(df["label"] == label) & (df["id"] == i)]

            if len(tmp) == 0:
                continue

            epochs = tmp["epoch"].values
            meas_times = tmp["meas_time"].values

            epoch_prev = epochs[0]
            for (epoch, meas_time) in zip(epochs[1:], meas_times[:-1]):
                if epoch - epoch_prev == 1:
                    # 読み込み、新規ファイル生成
                    seq_prev = np.load(f"{DIR_ARRAYS}/{i}_{epoch_prev}.npy")
                    seq_curr = np.load(f"{DIR_ARRAYS}/{i}_{epoch}.npy")
                    seq = np.concatenate([seq_prev, seq_curr])
                    for j in range(1, 3):
                        idx_start = 1000 * j
                        seq_dst = seq[idx_start : idx_start + 3000]
                        fname = f"{DIR_ARRAYS}/{i}_{epoch_prev}_{j}.npy"
                        np.save(fname, seq_dst)
                        meas_time_cross = meas_time + pd.Timedelta(seconds=10 * j)
                        data_cross.append([f"{epoch_prev}_{j}", meas_time_cross, label, i])
                epoch_prev = epoch

    df_cross = pd.DataFrame(data_cross, columns=["epoch", "meas_time", "label", "id"])
    df = pd.concat([df, df_cross], axis=0)
    df = df.groupby("id").apply(lambda x: x.sort_values("meas_time"))
    df.to_csv(f"{DIR_PROCESSED}/train_df_augument.csv", index=False)

    df = pd.read_csv(f"{DIR_PROCESSED}/train_df_augument.csv")
    folds = create_folds(df)
    folds.to_csv(f"{DIR_PROCESSED}/train_df_augunemt_fold.csv", index=False)


if __name__ == "__main__":
    main()
