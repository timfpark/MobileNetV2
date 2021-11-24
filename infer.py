import cv2
import os
import tensorflow as tf

CATEGORIES = ["nothing", "person"]

def prepare(file):
    IMG_SIZE = 244
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")

raw_directory = "raw"
for filename in os.listdir(raw_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(raw_directory, filename)

        image = prepare(image_path)
        prediction = model.predict([image])
        prediction = list(prediction[0])

        max_confidence = max(prediction)
        predicted_category = CATEGORIES[prediction.index(max(prediction))]

        new_path = os.path.join(raw_directory, predicted_category, filename)

        print("# " + str(max_confidence))
        print("mv " + image_path + " " + new_path)
