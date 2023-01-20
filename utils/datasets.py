import numpy as np
import torch
from torch.utils.data import Dataset

from code_factory.seq_aug import (AddGaussianNoise, GaussianNoiseSNR,
                                  PinkNoiseSNR, RandomCrop, TimeShift,
                                  TimeStretch, VolumeControl)
from parameters.data_path import DIR_ARRAYS


class TrainDataset(Dataset):
    def __init__(self, df, CFG):
        self.df = df
        self.CFG = CFG
        self.meta_cols = ["night", "age", "sex", "sleep_duration", "label_before"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df["id"].values[idx]
        epoch = self.df["epoch"].values[idx]
        label = self.df["label"].values[idx]
        seq = np.load(f"{DIR_ARRAYS}/{id_}_{epoch}.npy")
        # seq = seq[:, :2]

        if self.CFG.augmentation.crop_len > 0:
            l_ = np.random.randint(self.CFG.augmentation.crop_len)
            r = np.random.randint(self.CFG.augmentation.crop_len)
            seq[:l_, :] = 0
            seq[-r:, :] = 0
        if np.random.rand() < self.CFG.augmentation.add_gause:
            transform = AddGaussianNoise(always_apply=True, max_noise_amplitude=self.CFG.augmentation.max_noise_amplitude)
            seq = transform(seq)
        if np.random.rand() < self.CFG.augmentation.add_g_snr:
            transform = GaussianNoiseSNR(always_apply=True, min_snr=self.CFG.augmentation.min_snr, max_snr=self.CFG.augmentation.max_snr)
            seq = transform(seq)
        if np.random.rand() < self.CFG.augmentation.add_p_snr:
            transform = PinkNoiseSNR(always_apply=True, min_snr=self.CFG.augmentation.min_snr, max_snr=self.CFG.augmentation.max_snr)
            seq = transform(seq)
        if np.random.rand() < self.CFG.augmentation.t_strech:
            transform = TimeStretch(always_apply=True, max_rate=2.0)
            seq = transform(seq)
            if seq.shape[0] < self.CFG.augmentation.crop_total:
                seq = np.concatenate([seq, seq])
            if seq.shape[0] > self.CFG.augmentation.crop_total:
                trans = RandomCrop(height=self.CFG.augmentation.crop_total, width=12)
                seq = trans(image=seq)["image"]
        if np.random.rand() < self.CFG.augmentation.t_shift:
            transform = TimeShift(always_apply=True, max_shift_second=0.5, sr=100, padding_mode=self.CFG.augmentation.t_shift_mode)
            seq = transform(seq)
        if np.random.rand() < self.CFG.augmentation.v_cont:
            transform = VolumeControl(always_apply=True, mode=self.CFG.augmentation.v_mode)
            seq = transform(seq)
        seq = torch.from_numpy(seq).float()
        label = torch.tensor(label).long()

        if self.CFG.metadata.use is False:
            return seq, label
        else:
            metadata = self.df[self.meta_cols].values[idx]
            metadata = torch.tensor(metadata)
            return [seq, metadata], label


class TestDataset(Dataset):
    def __init__(self, df, CFG):
        self.df = df
        self.CFG = CFG
        self.meta_cols = ["night", "age", "sex", "sleep_duration"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df["id"].values[idx]
        epoch = self.df["epoch"].values[idx]
        seq = np.load(f"{DIR_ARRAYS}/{id_}_{epoch}.npy")
        seq = torch.from_numpy(seq).float()
        if self.CFG.metadata.use is False:
            return seq
        else:
            metadata = self.df[self.meta_cols].values[idx]
            metadata = torch.tensor(metadata)
            epoch = self.df["epoch"].values[idx]
            return [seq, metadata, epoch]
