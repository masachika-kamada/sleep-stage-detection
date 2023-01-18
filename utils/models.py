import timm
import torch
import torch.nn as nn

from utils.conv_stem import stft_conv, stft_conv_more, cwt_conv


def efficientnet(CFG):
    model = timm.create_model(CFG.model.name, pretrained=True, in_chans=7, num_classes=5)
    if CFG.model.conv_stem == "stft_conv":
        model.conv_stem = stft_conv(CFG)
    elif CFG.model.conv_stem == "stft_conv_more":
        model.conv_stem = stft_conv_more(CFG)
    elif CFG.model.conv_stem == "cwt_conv":
        model.conv_stem = cwt_conv(CFG)
    return model


class efficientnet_with_metadata(nn.Module):
    def __init__(self, CFG, n_meta_features=0, out_dim=5, n_meta_dim=[512, 128]):
        super(efficientnet_with_metadata, self).__init__()
        self.n_meta_features = n_meta_features
        self.CFG = CFG
        self.efficientnet = efficientnet(CFG)
        self.efficientnet.classifier = torch.nn.Identity()
        self.cnn_features_size = {"efficientnet_b0": 1280}

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.cnn_features_size["efficientnet_b0"]
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                nn.ReLU(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)  # catした特徴量を入れる

    def forward(self, x, x_meta=None):
        x = self.efficientnet(x)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out
