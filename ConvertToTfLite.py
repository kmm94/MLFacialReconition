import tensorflow as tf

modelName = "RGB_InceptionV3_Huber_loss"
path_To_Model="./savedModels/{}.h5".format(modelName)
with open(path_To_Model) as f:
    print("Succes")
model = tf.keras.models.load_model(path_To_Model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

Path_To_Save_Tflite = "./TfLiteModels/{}.tflite".format(modelName)
with open(Path_To_Save_Tflite, "wb") as file:
    file.write(tflite_model)

print("Model saved")