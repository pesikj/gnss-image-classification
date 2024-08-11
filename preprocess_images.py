import os
import random
import shutil

from PIL import Image


TOP_DIRECTORY = "data"
TOP_TARGET_DIRECTORY = "data_processed"
GROUP_COUNT = 5

def preprocess_images():
    if os.path.exists(TOP_TARGET_DIRECTORY):
        shutil.rmtree(TOP_TARGET_DIRECTORY)
    for bottom_directory in os.listdir(TOP_DIRECTORY):
        if not os.path.isdir(os.path.join(TOP_DIRECTORY, bottom_directory)):
            continue
        for file in os.listdir(os.path.join(TOP_DIRECTORY, bottom_directory)):
            file_path = os.path.join(TOP_DIRECTORY, bottom_directory, file)
            im = Image.open(file_path)
            width, height = im.size
            # left = 448
            # top = 38
            # right = left + 344
            # bottom = top + 289
            # im1 = im.crop((left, top, right, bottom))

            sub_directory = random.randint(1, GROUP_COUNT)
            target_directory = os.path.join(TOP_TARGET_DIRECTORY, str(sub_directory), bottom_directory)
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)
            target_file_path = os.path.join(TOP_TARGET_DIRECTORY, str(sub_directory), bottom_directory, file)
            im.save(target_file_path)


if __name__ == "__main__":
    preprocess_images()
