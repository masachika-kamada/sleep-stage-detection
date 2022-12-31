import gc
import logging
import math
import os
import random
import time
from contextlib import contextmanager

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.metrics as metrics
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from code_factory.pooling import AdaptiveConcatPool1d, AdaptiveConcatPool2d, GeM
from code_factory.seq_aug import Compose, AddGaussianNoise, GaussianNoiseSNR, TimeShift, TimeStretch
from code_factory.torch_cwt import CWT
from omegaconf import DictConfig, OmegaConf
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

from parameters.data_path import DIR_INPUT, DIR_OUTPUT, DIR_ARRAYS, DIR_PROCESSED


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.6f} s")


def time_function_execution(function_to_execute):
    def compute_execution_time(*args, **kwargs):
        start_time = time.time()
        result = function_to_execute(*args, **kwargs)
        end_time = time.time()
        computation_time = end_time - start_time
        print("Computation lasted: {}".format(computation_time))
        return result

    return compute_execution_time


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def bandpass(x, lf=20, hf=500, order=8, sr=1024):
    """
    Cell 33 of https://www.gw-openscience.org/LVT151012data/LOSC_Event_tutorial_LVT151012.html
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    window = signal.tukey(3000, 0.1)[:, None]
    if x.ndim == 2:
        x *= window
        for i in range(7):
            x[:, i] = signal.sosfilt(sos, x[:, i]) * normalization
    elif x.ndim == 3:  # batch
        for i in range(x.shape[0]):
            x[i] *= window
            for j in range(7):
                x[i, j] = signal.sosfilt(sos, x[i, j]) * normalization
    return x


def denoise_channel(ts, bandpass=[1, 40], signal_freq=100, bound=0.000125):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1

    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts[:, :2])

    ts_out[ts_out > bound] = bound
    ts_out[ts_out < -bound] = -bound
    ts[:, :2] = np.array(ts_out)
    return ts


class TrainDataset(Dataset):
    def __init__(self, df, CFG, train=True):
        self.df = df
        self.CFG = CFG

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df["id"].values[idx]
        epoch = self.df["epoch"].values[idx]
        label = self.df["label"].values[idx]
        seq = np.load(f"{DIR_ARRAYS}/{id_}_{epoch}.npy")  # 3000,7
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

        return seq, label


class TestDataset(Dataset):
    def __init__(self, df, CFG, train=True, transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        self.CFG = CFG

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df["id"].values[idx]
        epoch = self.df["epoch"].values[idx]
        seq = np.load(f"{DIR_ARRAYS}/{id_}_{epoch}.npy")  # 3000,7
        seq = torch.from_numpy(seq).float()
        return seq


def inference(model, test_loader, device, CFG):
    model.to(device)
    model.eval()
    probs = []
    # scaler = torch.cuda.amp.GradScaler()
    softmax = nn.Softmax(dim=1)

    for i, images in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
            y_preds = softmax(y_preds)
        probs.append(y_preds.to("cpu").numpy())

    probs = np.concatenate(probs)
    return probs


def cropaug(seq, crop_len):
    l_ = np.random.randint(crop_len)
    r_ = np.random.randint(crop_len)
    seq[:, :l_, :] = 0
    seq[:, -r_:, :] = 0
    return seq


def inference_tta(model, test_loader, device, CFG):
    model.to(device)
    model.eval()
    probs = []

    for i, images in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            y_pred = []
            for i in range(2):
                if i == 0:
                    augmented_image = images
                else:
                    augmented_image = cropaug(images, crop_len=CFG.augmentation.crop_len)

                y_preds, _ = model(augmented_image)
                softmax = nn.Softmax(dim=1)
                y_preds = softmax(y_preds)
                y_pred.append(y_preds.to("cpu").numpy())
            y_pred = np.mean(np.stack(y_pred, axis=0), axis=0)
        probs.append(y_pred)

    probs = np.concatenate(probs)
    return probs


SEQ_POOLING = {"gem": GeM(dim=2), "concat": AdaptiveConcatPool2d(), "avg": nn.AdaptiveAvgPool2d(1), "max": nn.AdaptiveMaxPool2d(1)}


class Noop(nn.Module):
    def __init__(self, *args):
        super(Noop, self).__init__()

    def forward(self, x):
        return x


class NoopAddDim(nn.Module):
    def __init__(self):
        super(NoopAddDim, self).__init__()

    def forward(self, x):
        return x.unsqueeze(-1)


def add_to_dim(x, num_dims, dim=0):
    while len(x.shape) < num_dims:
        x = x.unsqueeze(dim)
    return x


class DummyEmbd(nn.Module):
    def __init__(self, out_size, dtype=torch.float32):
        super(DummyEmbd, self).__init__()
        self.out_size = out_size
        self.dtype = dtype

    def forward(self, x):
        return torch.zeros(x.shape + (self.out_size,), dtype=self.dtype, device=x.device)


def soft_cross_entropy(input, target):
    return -(target * F.log_softmax(input, dim=1)).sum(1).mean(0)


def calc_positional_encoder(d_model, max_seq_len=32):
    # create constant 'pe' matrix with values dependant on
    # pos and i
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return pe / (d_model**0.5)


class TransformerModel(nn.Module):
    def __init__(
        self,
        in_size=3,  # d_model
        dim_feedforward=16,  # hidden
        n_heads=4,
        n_encoders=4,
        num_outputs=8,
        use_age=False,
        max_site_num=7,
        use_sex=False,
        use_age_diff=False,
        use_position_enc=False,
        pool="avg",
        embed=True,
    ):
        super(TransformerModel, self).__init__()
        self.in_size = in_size
        self.do_embed = embed
        self.encoder_layer = nn.TransformerEncoderLayer(in_size, n_heads, dim_feedforward=dim_feedforward)  # d_model(単語の次元..1280)
        #        self.decoder_layer =nn.TransformerDecoderLayer(in_size, 4, dim_feedforward=in_size)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_encoders)
        #        self.decoder=nn.TransformerDecoder(self.decoder_layer, 2)
        self.egg_emb = nn.Sequential(nn.Linear(7, 16), nn.ReLU(), nn.Linear(16, in_size))
        # meta_feature(csvから取ってくるやつ) ==========================
        self.sex_embd = nn.Embedding(2, in_size) if use_sex else DummyEmbd(in_size)
        self.age_embd = nn.Sequential(NoopAddDim(), nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, in_size)) if use_age else DummyEmbd(in_size)
        self.site_embd = nn.Embedding(max_site_num, in_size) if max_site_num > 0 else DummyEmbd(in_size)
        self.age_diff_embd = nn.Sequential(NoopAddDim(), nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, in_size)) if use_age else DummyEmbd(in_size)
        # ==========================
        if pool in ("avg", "concat", "gem", "max"):
            self.pool = SEQ_POOLING[pool]
            if pool == "concat":
                self.exam_classifier = nn.Linear(in_size * 2, num_outputs)
            else:
                self.exam_classifier = nn.Linear(in_size, num_outputs)
        else:
            self.pool = None
            self.exam_classifier = nn.Linear(in_size, num_outputs)
        # self.image_classifier = nn.Linear(in_size, num_outputs)
        self.pos_embd = calc_positional_encoder(7, max_seq_len=700) if use_position_enc else None
        self.layer1 = nn.Sequential(
            nn.Conv1d(12, in_size, kernel_size=13, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(in_size),
            # nn.ReLU(inplace=True),
            Mish(),
            nn.Conv1d(in_size, in_size * 2, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(in_size * 2),
            # nn.ReLU(inplace=True),
            Mish(),
            nn.Conv1d(in_size * 2, in_size, kernel_size=7, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(in_size),
            # nn.ReLU(inplace=True),
            Mish(),
            # SEModule(dim, dim//4),
        )
        self.gru = nn.GRU(7, in_size // 2, bidirectional=True, batch_first=True, num_layers=2)
        self.gru_afetr = nn.GRU(in_size, in_size, bidirectional=False, batch_first=True, num_layers=2)

    # @time_function_execution
    def forward(self, x, sex=None, age=None, site=None, age_diff=None, mask=None):
        if self.pos_embd is not None:
            if self.pos_embd.device != x.device:
                self.pos_embd = self.pos_embd.to(x.device)
        # meta_feature(csvから取ってくるやつ) ==========================...yuvalのsiimより??
        x = x if sex is None else x + self.sex_embd(sex)
        x = x if age is None else x + self.age_embd(age)
        x = x if site is None else x + self.site_embd(site)
        x = x if age_diff is None else x + self.age_diff_embd(age_diff)
        # ==========================
        # print(self.pos_embd[:x.shape[1]][None].shape,x.shape)
        x = x if self.pos_embd is None else x + self.pos_embd[: x.shape[1]][None]
        x = x if mask is None else x * mask.unsqueeze(-1)
        # x = x if self.do_embed ==False else self.layer1(x.permute(0, 2, 1)).permute(0, 2, 1)#self.egg_emb(x)
        x = x if self.do_embed is False else self.gru(x)[0]  # self.egg_emb(x)
        x = self.encoder(x)  # ,mask=s_mask #bs,len,depth
        # x = self.gru_afetr(x)[0]
        if self.pool:
            pooled = self.pool(x.permute(0, 2, 1))
            feature = pooled[:, :, 0]
            exam = self.exam_classifier(feature)
        else:
            feature = x[:, 0, :]
            exam = self.exam_classifier(feature)  # 0番目の"単語"にクラス分類のための特徴が集約されるように学習
        # image = self.image_classifier(x[:,:,:])
        return exam, feature


SEQ_POOLING = {"gem": GeM(dim=1), "concat": AdaptiveConcatPool1d(), "avg": nn.AdaptiveAvgPool1d(1), "max": nn.AdaptiveMaxPool1d(1)}


class Mish(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh(F.softplus(input))


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


# wave_net
# https://www.kaggle.com/cswwp347724/wavenet-pytorch/#data
class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(
                    out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate
                )
            )
            self.gate_convs.append(
                nn.Conv1d(
                    out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate
                )
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class Classifier(nn.Module):
    def __init__(self, inch=12, kernel_size=3, num_classes=2, pool="avg"):
        super().__init__()
        # self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)

        self.pool = nn.AdaptiveAvgPool1d(1)  # SEQ_POOLING[pool]
        dim = 32
        self.fc = nn.Linear(128, 2)
        self.head = nn.Sequential(nn.Linear(128, dim), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(dim, num_classes))
        self.flatten = nn.Flatten()

    def forward(self, x):
        # (batch_size,seq_len,depth)
        x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        # x = x.permute(0, 2, 1)
        # x, _ = self.LSTM(x)
        x = self.pool(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc(x)  # bs,2

        return x


def get_transforms1(*, data, CFG):
    if data == "train":
        return Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomContrast(p=0.5),
                GaussNoise(p=0.5),
                RandomRotate90(p=0.5),
                # RandomGamma(p=0.5),
                RandomBrightnessContrast(p=0.5),
                GaussianBlur(p=0.5),
                # Cutout(p=CFG.augmentation.cutout_p),
                # Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )
    elif data == "valid":
        return Compose(
            [
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ]
        )


def AUC(y_true, y_pred, onehot=False):
    if y_pred.shape[1] == 2:
        auc = roc_auc_score(y_true, y_pred[:, 1])
    else:
        if onehot:
            auc = 0
            n_col = y_pred.shape[1]
            print(n_col, y_true[:, i].shape)
            for i in range(n_col):
                auc += roc_auc_score(y_true[:, i], y_pred[:, i]) / n_col

        else:
            auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
    return auc


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.shape[0]  # bs,seq_len,depth
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


kernel_dict = {
    "tf_efficientnet_b0_ns": 32,
    "tf_efficientnet_b2_ns": 32,
    "tf_efficientnetv2_s": 24,
    "tf_efficientnetv2_m": 24,
    "tf_efficientnet_b5_ns": 48,
    "tf_efficientnetv2_l": 32,
    "convnext_nano": 80,
    "convnext_tiny": 96,
    "convnext_small": 96,
}


class stft_conv(nn.Module):
    def __init__(self, CFG):
        super(stft_conv, self).__init__()
        self.conv = nn.Conv2d(7, kernel_dict[CFG.model.name], kernel_size=3, padding=1, stride=1, bias=False)
        self.n_fft = 128  # 64 not work 256 not work
        # self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=CFG.augmentation.specaug_time)
        # self.frec_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=CFG.augmentation.specaug_frec)
        self.CFG = CFG

    def forward(self, x):
        x = self.torch_stft(x)  # 128:(7, 65, 180) 256:(7, 129, 86) 210:(7, 106, 108)
        # batch_size = x.shape[0]
        # if self.training:
        #     index = torch.randperm(batch_size).cuda()
        #     if self.CFG.augmentation.specaug_time>0:
        #        x[index[len(index)//2:]]=self.time_mask(x[index[len(index)//2:]])
        #     if self.CFG.augmentation.specaug_frec>0:
        #        x[index[:len(index)//2]]=self.frec_mask(x[index[:len(index)//2]])
        x = self.conv(x)
        return x

    def torch_stft(self, X_train):
        signal = []

        # index = np.arange(X_train.shape[1])
        # np.random.shuffle(index)
        for s in range(X_train.shape[-1]):
            spectral = torch.stft(X_train[:, :, s], n_fft=self.n_fft, hop_length=self.n_fft * 1 // 4, center=False, onesided=True)
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=-1)


class stft_conv_nfnet(nn.Module):
    # not work
    def __init__(self, CFG):
        super(stft_conv_nfnet, self).__init__()
        self.conv = timm.models.layers.ScaledStdConv2d(7, 16, kernel_size=3, padding=1, stride=1, bias=False)
        self.n_fft = 128

    def forward(self, x):
        x = self.torch_stft(x)  # 128:(7, 65, 180) 256:(7, 129, 86) 210:(7, 106, 108)
        # specaug

        x = self.conv(x)
        return x

    def torch_stft(self, X_train):
        signal = []

        # index = np.arange(X_train.shape[1])
        # np.random.shuffle(index)
        for s in range(X_train.shape[-1]):
            spectral = torch.stft(X_train[:, :, s], n_fft=self.n_fft, hop_length=self.n_fft * 1 // 4, center=False, onesided=True)
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=-1)


class stft_conv_more(nn.Module):
    def __init__(self, CFG):
        super(stft_conv_more, self).__init__()
        """
        SETI alaskaのqishen ha model....convを重ねて高解像度の情報をより抽出する
        """
        self.qishen_conv = nn.Sequential(
            nn.Conv2d(7, 12, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.Mish(),
            nn.Conv2d(12, 24, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Mish(),
        )
        self.conv = nn.Conv2d(24, kernel_dict[CFG.model.name], kernel_size=3, padding=1, stride=1, bias=False)
        self.n_fft = 128

    def forward(self, x):
        x = self.torch_stft(x)

        x = self.qishen_conv(x)
        x = self.conv(x)
        return x

    def torch_stft(self, X_train):
        signal = []

        # index = np.arange(X_train.shape[1])
        # np.random.shuffle(index)
        for s in range(X_train.shape[-1]):
            spectral = torch.stft(X_train[:, :, s], n_fft=self.n_fft, hop_length=self.n_fft * 1 // 4, center=False, onesided=True)
            signal.append(spectral)
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=-1)


class stft_conv_224(nn.Module):
    def __init__(self, CFG):
        super(stft_conv_224, self).__init__()
        """
        ViT系のために224*224にresize
        """
        self.conv = nn.Conv2d(7, 32, kernel_size=3, padding=1, stride=1, bias=False)
        self.n_fft = 210  # 128*2

    def forward(self, x):
        x = self.torch_stft(x)
        x = torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(x)
        x = self.conv(x)
        return x

    def torch_stft(self, X_train):
        signal = []

        # index = np.arange(X_train.shape[1])
        # np.random.shuffle(index)
        for s in range(X_train.shape[-1]):
            spectral = torch.stft(X_train[:, :, s], n_fft=self.n_fft, hop_length=self.n_fft * 1 // 4, center=False, onesided=True)
            signal.append(spectral)
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=-1)


class cwt_conv(nn.Module):
    def __init__(self, CFG):
        super(cwt_conv, self).__init__()
        self.conv = nn.Conv2d(7, kernel_dict[CFG.model.name], kernel_size=3, padding=1, stride=1, bias=False)
        self.n_fft = 128 * 2
        self.cwt = CWT(dt=1 / 512, dj=0.0625)

    def forward(self, x):
        x = self.cwt(x.permute(0, 2, 1))
        print(x.shape)
        exit()
        x = self.conv(x)
        return x


def train_fn(CFG, fold, folds):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"### fold: {fold} ###")
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index
    train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True), train=True, CFG=CFG)  #
    val_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True), train=False, CFG=CFG)  #

    train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size * 2, shuffle=False, num_workers=8)

    # === model select ===
    # STFTしてから2d cnn
    model = timm.create_model(CFG.model.name, pretrained=True, in_chans=7, num_classes=5)
    print(model.conv_stem)

    model.conv_stem = stft_conv(CFG)  # stft_conv_more(CFG)
    # model.conv_stem = cwt_conv(CFG)
    print(model.conv_stem)
    model.to(device)
    print(model.conv_stem)
    # ============

    # === optim select ===
    if CFG.train.optim == "adam":
        optimizer = AdamW(model.parameters(), lr=CFG.train.lr, amsgrad=False)  # CFG.train.lr
    # ============

    # === scheduler select ===
    if CFG.train.scheduler.name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.train.epochs, eta_min=CFG.train.scheduler.min_lr)
    elif CFG.train.scheduler.name == "cosine_warm":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.train.scheduler.t_0, T_mult=1, eta_min=CFG.train.scheduler.min_lr, last_epoch=-1)
    elif CFG.train.scheduler.name == "reduce":
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2, verbose=True, eps=1e-6)
    # ============
    # === loss select ===
    if CFG.loss.name == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG.loss.smooth_a)
    elif CFG.loss.name == "focal":
        criterion = FocalLoss_CE(alpha=1, gamma=CFG.loss.focal_gamma)
    # ============
    best_score = 0
    best_preds = None

    softmax = nn.Softmax(dim=1)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(CFG.train.epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.0

        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in tk0:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            rand = np.random.rand()
            if CFG.mixup > rand:
                images, y_a, y_b, lam = mixup_data(images, labels, alpha=2)
            with torch.cuda.amp.autocast():
                y_preds = model(images.float())
                if CFG.mixup > rand:
                    loss = mixup_criterion(criterion, y_preds, y_a, y_b, lam)
                else:
                    loss = criterion(y_preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if CFG.train.scheduler.name != "none":
            scheduler.step()

            avg_loss += loss.item() / len(train_loader)
        model.eval()
        avg_val_loss = 0.0
        preds = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, labels) in tk1:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                y_preds = model(images.float())
                loss = criterion(y_preds, labels)
            valid_labels.append(labels.to("cpu").numpy())
            y_preds = softmax(y_preds)
            preds.append(y_preds.to("cpu").detach().numpy())
            avg_val_loss += loss.item() / len(valid_loader)
        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)

        score = AUC(valid_labels, preds)

        elapsed = time.time() - start_time
        log.info(f"  Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f}  time: {elapsed:.0f}s")
        log.info(f"  Epoch {epoch+1} - AUC: {score:.6f}")
        th_preds = np.argmax(preds, axis=1)
        acc = accuracy_score(valid_labels, th_preds)
        log.info(f"  Epoch {epoch+1} - Acc: {acc:.4f}")
        if score > best_score:
            best_score = score
            best_preds = preds
            log.info(f"  Epoch {epoch+1} - Save Best Score: {best_score:.4f}")
            torch.save(model.state_dict(), f"{DIR_OUTPUT}/weights/fold{fold}_{CFG.general.exp_num}_baseline.pth")
        for i in range(5):
            col = f"pred_{i}"
            val_folds[col] = best_preds[:, i]
    return best_preds, valid_labels, val_folds


def submit(test, CFG):
    print("run inference")
    torch.cuda.set_device(CFG.general.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = TestDataset(test, CFG=CFG)
    test_loader = DataLoader(test_dataset, batch_size=CFG.train.batch_size * 2, shuffle=False)
    probs = []
    # probs_rank = []
    for fold in range(5):
        weights_path = f"{DIR_OUTPUT}/weights/fold{fold}_{CFG.general.exp_num}_baseline.pth"
        if CFG.model.name == "trans":
            model = TransformerModel(in_size=32, dim_feedforward=64, num_outputs=2, n_heads=1, n_encoders=2, pool="avg", embed=True)
        elif CFG.model.name == "gru":
            model = gru_model(input_ch=7, pool="avg")
        else:
            model = CNN_1d(num_classes=2, input_ch=12, pool=CFG.model.pooling, dim=CFG.model.hidden_dim)
        model = timm.create_model(CFG.model.name, pretrained=True, num_classes=5)
        model.conv_stem = stft_conv(CFG)  # stft_conv_more(CFG)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

        _probs = inference(model, test_loader, device, CFG)
        probs.append(_probs)
    probs = np.mean(probs, axis=0)
    return probs


config_path = "parameters/"
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=config_path, config_name="config.yaml")
def main(CFG: DictConfig) -> None:

    seed_torch(seed=42)
    log.info(f"===============exp_num: {CFG.general.exp_num}============")

    folds = pd.read_csv(f"{DIR_PROCESSED}/train_df_fold.csv")
    tmp = folds[folds["id"] == "5edb9d9"]
    tmp = tmp[tmp["epoch"] == 1101]
    folds = folds.drop(index=tmp.index).reset_index(drop=True)
    test = pd.read_csv(f"{DIR_PROCESSED}/test_df0.csv")

    preds = []
    valid_labels = []

    oof = pd.DataFrame()
    for fold in range(5):
        if CFG.general.debug:
            fold = 4
        _preds, _valid_labels, _val_oof = train_fn(CFG, fold, folds)
        preds.append(_preds)
        valid_labels.append(_valid_labels)
        oof = pd.concat([oof, _val_oof])

    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)
    th_preds = np.argmax(preds, axis=1)

    acc = classification_report(valid_labels, th_preds)
    log.info(f"  =====Acc====== \n{acc}")

    cm = confusion_matrix(valid_labels, th_preds)
    log.info(f"  =====Acc====== \n{cm}")

    score = AUC(valid_labels, preds)
    log.info(f"  =====AUC(CV)====== \n{score}")
    oof.to_csv(f"{DIR_OUTPUT}/oof_{CFG.general.exp_num}.csv", index=False)

    pred = submit(test, CFG)

    for i in range(5):
        col = f"pred_{i}"
        test[col] = pred[:, i]
    test.to_csv(f"{DIR_OUTPUT}/predict_{CFG.general.exp_num}.csv", index=False)
    th_pred = np.argmax(pred, axis=1)
    submission = pd.read_csv(f"{DIR_INPUT}/sample_submission.csv")
    dic = {"Sleep stage 3/4": 3, "Sleep stage 2": 2, "Sleep stage W": 0, "Sleep stage R": 4, "Sleep stage 1": 1}

    def get_swap_dict(d):
        return {v: k for k, v in d.items()}

    d_swap = get_swap_dict(dic)

    def func2(x):
        return d_swap[x]

    submission["condition"] = th_pred
    submission["condition"] = submission["condition"].apply(func2)

    submission.to_csv(f"{DIR_OUTPUT}/submit_{CFG.general.exp_num}.csv", index=False)


if __name__ == "__main__":
    main()
