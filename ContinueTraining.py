import tensorflow as tf
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, CSVLogger
import DataManager as dm

filepath = "checkpoints/checkpointBlackAndWhite-{epoch:04d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='auto')


csv_fileName = "logs/CSV_log_BlackAndWhite.csv"
logger = CSVLogger(
    csv_fileName, separator=',', append=False
)

npImgArray, npLabelArray = dm.getImgAndLables()

model = tf.keras.models.load_model("checkpoints/checkpointBlackAndWhite-0216.h5")

model.fit(npImgArray, npLabelArray, epochs=10, batch_size=5, validation_split=0.2, callbacks=[checkpoint, logger])

model.save("./savedModels/myModelBlackAndWhite.h5")