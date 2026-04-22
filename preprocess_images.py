import os
import random
import re
import shutil

import pandas as pd
from PIL import Image


TOP_DIRECTORY = "data"
TOP_TARGET_DIRECTORY = "data_processed"
GROUP_COUNT = 4


def preprocess_images() -> None:
    if os.path.exists(TOP_TARGET_DIRECTORY):
        shutil.rmtree(TOP_TARGET_DIRECTORY)

    category_dirs = [
        x for x in os.listdir(TOP_DIRECTORY)
        if re.fullmatch(r"Kategorie +\d +- +[\w() ]*", x)
    ]

    for i in range(GROUP_COUNT):
        for cat in category_dirs:
            os.makedirs(os.path.join(TOP_TARGET_DIRECTORY, str(i + 1), cat))

    files_overview = []
    for cat in category_dirs:
        files = os.listdir(os.path.join(TOP_DIRECTORY, cat))
        random.shuffle(files)
        for i, file in enumerate(files):
            target_group = str(i % GROUP_COUNT + 1)
            if target_group == "1" and len(files) - i < GROUP_COUNT:
                break
            file_path = os.path.join(TOP_DIRECTORY, cat, file)
            im = Image.open(file_path)
            target_file_path = os.path.join(TOP_TARGET_DIRECTORY, target_group, cat, file)
            im.save(target_file_path)
            files_overview.append([cat, target_group, file])

    overview_df = pd.DataFrame(files_overview, columns=["Category", "Group", "File"])
    crosstab = pd.crosstab(overview_df["Category"], overview_df["Group"])
    print(crosstab)


if __name__ == "__main__":
    preprocess_images()
