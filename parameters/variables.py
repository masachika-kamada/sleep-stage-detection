# Sleep stage 3とSleep stage 4を同じIDとして、AASMによる分類に変更する
RANDK_LABEL2ID = {
    "Movement time": -1,
    "Sleep stage ?": -1,
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}
# AASMによる分類
LABEL2ID = {
    "Movement time": -1,
    "Sleep stage ?": -1,
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3/4": 3,
    "Sleep stage R": 4
}
ID2LABEL = {v:k for k, v in LABEL2ID.items()}

COLS = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
    "Resp oro-nasal",
    "EMG submental",
    "Temp rectal",
    "Event marker"
]
