
import tensorflow as tf
from tensorflow_core.python.keras import Model, Input
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow_core.python.keras.optimizers import Adam


def unet(pretrained_weights=None, input_size=(320, 320, 3)):
    NNinput = tf.keras.layers.Input(input_size)
    conv1 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(NNinput)
    conv1 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(filters=128, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(filters=64, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(filters=32, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(filters=2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    flatten = tf.keras.layers.Flatten()(conv9)
    dense1 = tf.keras.layers.Dense(units=128, activation="relu")(flatten)
    outputLayer = tf.keras.layers.Dense(units=6)(dense1)

    model = tf.keras.models.Model(NNinput, outputLayer)

    #model.compile(optimizer=Adam(lr=1e-4), loss='mean_absolute_error', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model