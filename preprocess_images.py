import os
import random

from PIL import Image

top_directory = "data"
top_target_directory = "data_processed"
sub_directories = ["train", "val"]
for bottom_directory in os.listdir(top_directory):
    for file in os.listdir(os.path.join(top_directory, bottom_directory)):
        file_path = os.path.join(top_directory, bottom_directory, file)
        im = Image.open(file_path)
        width, height = im.size
        left = 448
        top = 38
        right = left + 344
        bottom = top + 289
        im1 = im.crop((left, top, right, bottom))

        sub_directory = sub_directories[1 if random.uniform(0, 1) > 0.7 else 0]
        target_directory = os.path.join(top_target_directory, sub_directory, bottom_directory)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        target_file_path = os.path.join(top_target_directory, sub_directory, bottom_directory, file)
        im1.save(target_file_path)

