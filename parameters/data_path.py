import os

DIR_INPUT = os.path.abspath("data/input")
DIR_EDF = f"{DIR_INPUT}/edf_data"
DIR_ARRAYS = os.path.abspath("data/arrays")
DIR_OUTPUT = os.path.abspath("data/output")
DIR_PROCESSED = os.path.abspath("data/processed")

# それぞれ存在していなかったら作成
for path in [DIR_INPUT, DIR_EDF, DIR_ARRAYS, DIR_OUTPUT, DIR_PROCESSED]:
    if not os.path.exists(path):
        os.makedirs(path)
