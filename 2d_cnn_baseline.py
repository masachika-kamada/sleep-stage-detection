import logging
import os
import random
import time

import hydra
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau)
from torch.utils.data import DataLoader
from tqdm import tqdm

from parameters.data_path import DIR_INPUT, DIR_OUTPUT, DIR_PROCESSED
from parameters.variables import ID2LABEL
from utils.conv_stem import stft_conv, stft_conv_more, cwt_conv
from utils.datasets import TestDataset, TrainDataset
from utils.metrics import AUC
from utils.mixup import mixup_data, mixup_criterion
from utils.loss_func import FocalLoss_CE


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def inference(model, test_loader, device, CFG):
    model.to(device)
    model.eval()
    probs = []
    softmax = nn.Softmax(dim=1)

    for i, images in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
            y_preds = softmax(y_preds)
        probs.append(y_preds.to("cpu").numpy())

    probs = np.concatenate(probs)
    return probs


def train_fn(CFG, fold, folds):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n### fold: {fold} ###\n")
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index
    train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True), train=True, CFG=CFG)
    val_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True), train=False, CFG=CFG)

    train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size * 2, shuffle=False, num_workers=8)

    # === model select ===
    model = timm.create_model(CFG.model.name, pretrained=True, in_chans=7, num_classes=5)
    model.conv_stem = stft_conv(CFG)
    # model.conv_stem = stft_conv_more(CFG)
    # model.conv_stem = cwt_conv(CFG)
    model.to(device)
    print(model.conv_stem)

    # === optim select ===
    if CFG.train.optim == "adam":
        optimizer = AdamW(model.parameters(), lr=CFG.train.lr, amsgrad=False)

    # === scheduler select ===
    if CFG.train.scheduler.name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.train.epochs, eta_min=CFG.train.scheduler.min_lr)
    elif CFG.train.scheduler.name == "cosine_warm":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.train.scheduler.t_0, T_mult=1, eta_min=CFG.train.scheduler.min_lr, last_epoch=-1)
    elif CFG.train.scheduler.name == "reduce":
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2, verbose=True, eps=1e-6)

    # === loss select ===
    if CFG.loss.name == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG.loss.smooth_a)
    elif CFG.loss.name == "focal":
        criterion = FocalLoss_CE(alpha=CFG.loss.focal_alpha, gamma=CFG.loss.focal_gamma)

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
            torch.save(model.state_dict(), f"{DIR_OUTPUT}/weights/fold{fold}_{CFG.general.exp_num}.pth")
        for i in range(5):
            col = f"pred_{i}"
            val_folds[col] = best_preds[:, i]
    return best_preds, valid_labels, val_folds


def pred_fn(test, CFG):
    print("run inference")
    torch.cuda.set_device(CFG.general.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = TestDataset(test, CFG=CFG)
    test_loader = DataLoader(test_dataset, batch_size=CFG.train.batch_size * 2, shuffle=False)
    probs = []
    for fold in range(5):
        weights_path = f"{DIR_OUTPUT}/weights/fold{fold}_{CFG.general.exp_num}.pth"
        model = timm.create_model(CFG.model.name, pretrained=True, num_classes=5)
        model.conv_stem = stft_conv(CFG)
        # model.conv_stem = stft_conv_more(CFG)
        # model.conv_stem = cwt_conv(CFG)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

        _probs = inference(model, test_loader, device, CFG)
        probs.append(_probs)
    probs = np.mean(probs, axis=0)
    return probs


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="parameters/", config_name="config.yaml")
def main(CFG: DictConfig) -> None:

    seed_torch(seed=CFG.general.seed)
    log.info(f"===== exp_num: {CFG.general.exp_num} =====")

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

    log.info(f"\n===== Acc =====\n{classification_report(valid_labels, th_preds)}")
    log.info(f"\n===== Acc =====\n{confusion_matrix(valid_labels, th_preds)}")
    log.info(f"\n=== AUC(CV) ===\n{AUC(valid_labels, preds)}")
    oof.to_csv(f"{DIR_OUTPUT}/oof_{CFG.general.exp_num}.csv", index=False)

    pred = pred_fn(test, CFG)

    for i in range(5):
        col = f"pred_{i}"
        test[col] = pred[:, i]
    test.to_csv(f"{DIR_OUTPUT}/predict_{CFG.general.exp_num}.csv", index=False)
    th_pred = np.argmax(pred, axis=1)

    submission = pd.read_csv(f"{DIR_INPUT}/sample_submission.csv")
    submission["condition"] = th_pred
    submission["condition"] = submission["condition"].map(ID2LABEL)
    submission.to_csv(f"{DIR_OUTPUT}/submit_{CFG.general.exp_num}.csv", index=False)


if __name__ == "__main__":
    main()
