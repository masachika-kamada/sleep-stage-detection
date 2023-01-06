import pandas as pd

from parameters.data_path import DIR_EDF, DIR_INPUT, DIR_PROCESSED
from utils.data_conversion import create_annoation_df
from utils.kfold import create_folds


def main():
    train_record_df = pd.read_csv(f"{DIR_INPUT}/train_records.csv")
    train_record_df["hypnogram"] = DIR_EDF + "/" + train_record_df["hypnogram"]
    train_record_df["psg"] = DIR_EDF + "/" + train_record_df["psg"]

    test_record_df = pd.read_csv(f"{DIR_INPUT}/test_records.csv")
    test_record_df["psg"] = DIR_EDF + "/" + test_record_df["psg"]

    train_df = create_annoation_df(train_record_df, is_test=False)
    train_df.to_csv(f"{DIR_PROCESSED}/train_df0.csv", index=False)
    # (161610, 4)
    test_df = create_annoation_df(test_record_df, is_test=True)
    test_df.to_csv(f"{DIR_PROCESSED}/test_df0.csv", index=False)
    # print(test_df.shape)  # (52296, 4)

    train_df = pd.read_csv(f"{DIR_PROCESSED}/train_df0.csv")

    folds = create_folds(train_df)
    folds.to_csv(f"{DIR_PROCESSED}/train_df_fold.csv", index=False)


if __name__ == "__main__":
    main()
