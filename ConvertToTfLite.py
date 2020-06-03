import tensorflow as tf


path_To_Model="savedModels\CNNv25_Marcin_DropOut04.h5" #"./savedModels/{}.h5".format(modelName)
with open(path_To_Model) as f:
    print("Succes")
model = tf.keras.models.load_model(path_To_Model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

Path_To_Save_Tflite = "./TfLiteModels/CNNv25_OUH_Dropout06.tflite"#.format(modelName)
with open(Path_To_Save_Tflite, "wb") as file:
    file.write(tflite_model)

print("Model saved")