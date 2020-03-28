import tensorflow as tf

path_To_Model="./savedModels/myModelBlackAndWhite.h5"
with open(path_To_Model) as f:
    print("Succes")
model = tf.keras.models.load_model(path_To_Model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

Path_To_Save_Tflite = "./TfLiteModels/myModelBlackAndWhite.tflite"
with open(Path_To_Save_Tflite, "wb") as file:
    file.write(tflite_model)

print("Model saved")