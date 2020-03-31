import tensorflow as tf




model = tf.keras.models.load_model("checkpoints/checkpointBlackAndWhite-0216.h5")

model.fit(npImgArray, npLabelArray, epochs=10, batch_size=5, validation_split=0.2, callbacks=[checkpoint, logger])