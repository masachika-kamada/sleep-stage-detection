import torch.nn as nn
import torch
import timm
import torchvision
from code_factory.torch_cwt import CWT
from parameters.variables import KERNEL_DICT


class stft_conv(nn.Module):
    def __init__(self, CFG):
        super(stft_conv, self).__init__()
        self.conv = nn.Conv2d(7, KERNEL_DICT[CFG.model.name], kernel_size=3, padding=1, stride=1, bias=False)
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
            spectral = torch.stft(X_train[:, :, s], n_fft=self.n_fft, hop_length=self.n_fft * 1 // 4,
                                  center=False, onesided=True, return_complex=False)
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
            spectral = torch.stft(X_train[:, :, s], n_fft=self.n_fft, hop_length=self.n_fft * 1 // 4,
                                  center=False, onesided=True, return_complex=False)
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
        self.conv = nn.Conv2d(24, KERNEL_DICT[CFG.model.name], kernel_size=3, padding=1, stride=1, bias=False)
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
            spectral = torch.stft(X_train[:, :, s], n_fft=self.n_fft, hop_length=self.n_fft * 1 // 4,
                                  center=False, onesided=True, return_complex=False)
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
            spectral = torch.stft(X_train[:, :, s], n_fft=self.n_fft, hop_length=self.n_fft * 1 // 4,
                                  center=False, onesided=True, return_complex=False)
            signal.append(spectral)
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=-1)


class cwt_conv(nn.Module):
    def __init__(self, CFG):
        super(cwt_conv, self).__init__()
        self.conv = nn.Conv2d(7, KERNEL_DICT[CFG.model.name], kernel_size=3, padding=1, stride=1, bias=False)
        self.n_fft = 128 * 2
        self.cwt = CWT(dt=1 / 512, dj=0.0625)

    def forward(self, x):
        x = self.cwt(x.permute(0, 2, 1))
        print(x.shape)
        exit()
        x = self.conv(x)
        return x
