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

# Method 1: Try loading with compile=False
try:
    best_model = tf.keras.models.load_model(
        "notebooks/models/best_model.h5",
        compile=False,
        safe_mode=False,
    )
except Exception as e:
    print(f"Method 1 failed: {e}")

    # Method 2: Load only weights if you have the model architecture
    try:
        # Recreate the model architecture (you'll need to match your original model)
        best_model = Sequential(
            [
                Input(shape=(244, 244, 3), name="input_layer"),
                # Wrap the base_model in a functional API to make it compatible with Sequential
                tf.keras.layers.Lambda(
                    lambda x: base_model(x), name="base_model_wrapper"
                ),
                # Add dropout right after base model to prevent overfitting
                Dropout(0.2, name="dropout_base"),
                BatchNormalization(name="bn_base"),
                # First dense layer with regularization
                Dense(
                    512,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    bias_regularizer=tf.keras.regularizers.l2(0.001),
                    name="dense_1",
                ),
                BatchNormalization(name="bn_1"),
                Dropout(0.5, name="dropout_1"),
                # Second dense layer with regularization
                Dense(
                    256,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    bias_regularizer=tf.keras.regularizers.l2(0.001),
                    name="dense_2",
                ),
                BatchNormalization(name="bn_2"),
                Dropout(0.4, name="dropout_2"),
                # Third dense layer
                Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    name="dense_3",
                ),
                BatchNormalization(name="bn_3"),
                Dropout(0.3, name="dropout_3"),
                # Output layer
                Dense(1, activation="sigmoid", name="output_layer"),
            ],
            name="breast_cancer_classifier",
        )
        best_model.load_weights("notebooks/models/best_model.h5")
    except Exception as e2:
        print(f"Method 2 failed: {e2}")

        # Method 3: Try loading with custom_objects
        try:
            best_model = tf.keras.models.load_model(
                "notebooks/models/best_model.h5",
                custom_objects=None,
                compile=False,
                safe_mode=False,
            )
        except Exception as e3:
            print(f"All methods failed. Error: {e3}")
            raise

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
