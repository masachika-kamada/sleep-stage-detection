import os

import pandas as pd

from parameters.data_path import DIR_ARRAYS, DIR_EDF, DIR_INPUT, DIR_PROCESSED
from utils.data_conversion import create_annoation_df


def main():
    train_record_df = pd.read_csv(f"{DIR_INPUT}/train_records.csv")
    train_record_df["hypnogram"] = DIR_EDF + "/" + train_record_df["hypnogram"]
    train_record_df["psg"] = DIR_EDF + "/" + train_record_df["psg"]

    test_record_df = pd.read_csv(f"{DIR_INPUT}/test_records.csv")
    test_record_df["psg"] = DIR_EDF + "/" + test_record_df["psg"]

    os.makedirs(DIR_ARRAYS, exist_ok=True)

    train_df = create_annoation_df(train_record_df, is_test=False)
    train_df.to_csv(f"{DIR_PROCESSED}/train_df0.csv", index=False)
    # (161610, 4)
    test_df = create_annoation_df(test_record_df, is_test=True)
    test_df.to_csv(f"{DIR_PROCESSED}/test_df0.csv", index=False)
    # print(test_df.shape)  # (52296, 4)


if __name__ == "__main__":
    main()
