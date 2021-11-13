import cv2
import os
import pathlib
import random

for root, dirs, files in os.walk('raw-train'):
    for file in files:
        source_path = os.path.join(root, file)

        training_validation_decision = random.randint(0, 100)
        if training_validation_decision < 20:
            dest_set = "validation"
        else:
            dest_set = "train"

        p = pathlib.Path(root)
        intermediate_path = pathlib.Path(*p.parts[1:])
        dest_dir = os.path.join(dest_set, intermediate_path)
        dest_path = os.path.join(dest_dir, file)

        print("resizing " + source_path + " to " + dest_path)

        img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)

        resized_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        cv2.imwrite(dest_path, resized_img)

