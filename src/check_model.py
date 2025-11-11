import tensorflow as tf

model = tf.keras.models.load_model("notebooks/models/best_model.keras", compile=False, safe_mode=False,)
model.summary()
