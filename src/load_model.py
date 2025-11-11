import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np


base_model = hub.KerasLayer(
    "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
    trainable=False,
    name="efficientnetv2-b0",
)


def load_model(model_path):
    print("Loading model from:", model_path)
    try:
        best_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "base_model": base_model,
                # "KerasLayer": hub.KerasLayer,
            },
            compile=False,
            safe_mode=False,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return best_model


# Alternative: Try loading the .keras file instead
# final_model = tf.keras.models.load_model(
#     "notebooks/models/Breast_Cancer_Histopathology_Image_Classification_model.keras",
#     safe_mode=False,
# )


def predict_image(img_path, IMAGE_SIZE):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = img / 255.0
    img = tf.expand_dims(img, 0)

    best_model = load_model("notebooks/models/best_model.keras")

    prediction = best_model.predict(img, verbose=0)

    return prediction[0][0]


if __name__ == "__main__":
    img_path = "data/raw/archive/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-21998CD/100X/SOB_B_F-14-21998CD-100-002.png"
    IMAGE_SIZE = 224
    prediction = predict_image(img_path, IMAGE_SIZE)
    print(prediction)

    if prediction > 0.5:
        print("Predicted class is: Benign")
    else:
        print("Predicted class is: Malignant")
