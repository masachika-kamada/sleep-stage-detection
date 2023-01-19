import pandas as pd

from parameters.data_path import DIR_INPUT, DIR_PROCESSED


def add_metadata(df, df_input):
    df_list = []
    for i in range(len(df_input)):
        check_id = df_input["id"].iloc[i]

        # df_trainのidがcheck_idと一致する行を抽出
        df_ = df.loc[df["id"] == check_id].copy()
        df_["night"] = df_input["night"][i]
        df_["age"] = df_input["age"][i]
        df_["sex"] = df_input["sex"][i]
        df_["sex"].replace({"male": 0, "female": 1}, inplace=True)

        # df_train_idに最初の測定からの経過時間を追加
        df_["meas_time"] = pd.to_datetime(df_["meas_time"])
        df_["sleep_duration"] = (df_["meas_time"] - df_["meas_time"].iloc[0]).dt.total_seconds() / 60
        df_["sleep_duration"] = df_["sleep_duration"].round(1)

        if "label" in df_.columns:
            df_["label_before"] = df_["label"].shift(1)
            df_["label_before"] = df_["label_before"].fillna(df_["label"])

        df_list.append(df_)

    df_meta = pd.concat(df_list)
    return df_meta


def main():
    df_train = pd.read_csv(f"{DIR_PROCESSED}/train_df_fold.csv")
    df_train_aug = pd.read_csv(f"{DIR_PROCESSED}/train_df_augment_fold.csv")
    df_train_input = pd.read_csv(f"{DIR_INPUT}/train_records.csv")
    df_test = pd.read_csv(f"{DIR_PROCESSED}/test_df0.csv")
    df_test_input = pd.read_csv(f"{DIR_INPUT}/test_records.csv")

    df_train_meta = add_metadata(df_train, df_train_input)
    df_train_meta.to_csv(f"{DIR_PROCESSED}/train_df_meta.csv", index=False)
    df_train_meta_aug = add_metadata(df_train_aug, df_train_input)
    df_train_meta_aug.to_csv(f"{DIR_PROCESSED}/train_df_augment_meta.csv", index=False)
    df_test_meta = add_metadata(df_test, df_test_input)
    df_test_meta.to_csv(f"{DIR_PROCESSED}/test_df_meta.csv", index=False)


if __name__ == "__main__":
    main()
