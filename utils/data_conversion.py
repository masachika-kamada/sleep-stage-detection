import datetime
import os

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from parameters.data_path import DIR_ARRAYS, DIR_INPUT
from parameters.variables import COLS, LABEL2ID, RANDK_LABEL2ID


def epoch_to_df(epoch) -> pd.DataFrame:
    truncate_start_point = epoch.info["temp"]["truncate_start_point"]
    df = epoch.to_data_frame(verbose=False)
    new_meas_date = epoch.info["meas_date"].replace(tzinfo=None) + datetime.timedelta(seconds=truncate_start_point)
    df["meas_time"] = pd.date_range(start=new_meas_date, periods=len(df), freq=pd.Timedelta(1 / 100, unit="s"))
    return df


def clean_df(df, id_, save):
    df = df[~df["condition"].isin(["Sleep stage ?", "Movement time"])]
    epochs = df["epoch"].unique()
    df_ = pd.DataFrame()
    df_["epoch"] = epochs
    start_times = []
    conditions = []
    for epoch in epochs:
        tmp = df[df["epoch"] == epoch].reset_index(drop=True)
        condition = tmp["condition"].values[0]
        start_time = tmp["meas_time"].values[0]
        start_times.append(start_time)
        conditions.append(condition)
        if not save:
            continue
        if os.path.exists(f"{DIR_ARRAYS}/{id_}_{epoch}.npy"):
            continue
        else:
            array = tmp[COLS].values
            np.save(f"{DIR_ARRAYS}/{id_}_{epoch}.npy", array)
    df_["meas_time"] = start_times
    df_["target"] = conditions
    df_["id"] = id_
    return df_


def create_annoation_df(record_df: pd.DataFrame, include=None, is_test=False, save=True) -> pd.DataFrame:
    list_epoch_df = []

    sample_submission_df = pd.read_csv(f"{DIR_INPUT}/sample_submission.csv")
    sample_submission_df["meas_time"] = pd.to_datetime(sample_submission_df["meas_time"])

    for _, row in tqdm(record_df.iterrows(), total=len(record_df)):
        # PSGファイルとHypnogram(アノテーションファイルを読み込む)
        psg_edf = mne.io.read_raw_edf(row["psg"], include=include, verbose=False)
        if not is_test:
            # 訓練データの場合
            annot = mne.read_annotations(row["hypnogram"])

            # 切り捨て
            truncate_start_point = 3600 * 5
            truncate_end_point = (len(psg_edf) / 100) - (3600 * 5)
            annot.crop(truncate_start_point, truncate_end_point, verbose=False)

            # アノテーションデータの切り捨て
            psg_edf.set_annotations(annot, emit_warning=False)
            events, _ = mne.events_from_annotations(psg_edf, event_id=RANDK_LABEL2ID, chunk_duration=30.0, verbose=False)

            event_id = LABEL2ID
        else:
            # テストデータの場合
            start_psg_date = psg_edf.info["meas_date"]
            start_psg_date = start_psg_date.replace(tzinfo=None)
            test_start_time = sample_submission_df[sample_submission_df["id"] == row["id"]]["meas_time"].min()
            test_end_time = sample_submission_df[sample_submission_df["id"] == row["id"]]["meas_time"].max()

            truncate_start_point = int((test_start_time - start_psg_date).total_seconds())
            truncate_end_point = int((test_end_time - start_psg_date).total_seconds()) + 30

            event_range = list(range(truncate_start_point, truncate_end_point, 30))
            events = np.zeros((len(event_range), 3), dtype=int)
            events[:, 0] = event_range
            events = events * 100

            event_id = {"Sleep stage W": 0}

        # 30秒毎に1epochとする
        tmax = 30.0 - 1.0 / psg_edf.info["sfreq"]
        epoch = mne.Epochs(raw=psg_edf, events=events, event_id=event_id, tmin=0, tmax=tmax, baseline=None, verbose=False, on_missing="ignore")

        # 途中でデータが落ちてないかチェック
        assert len(epoch.events) * 30 == truncate_end_point - truncate_start_point

        # メタデータを追加
        epoch.info["temp"] = {
            "id": row["id"],
            "subject_id": row["subject_id"],
            "night": row["night"],
            "age": row["age"],
            "sex": row["sex"],
            "truncate_start_point": truncate_start_point,
        }

        df_subset = clean_df(epoch_to_df(epoch), row["id"], save)
        list_epoch_df.append(df_subset)

    df = pd.concat(list_epoch_df).reset_index(drop=True)
    return df
