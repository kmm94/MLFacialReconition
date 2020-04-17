import tensorflow as tf
from tensorflow_core.python.keras import Model, Input
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Flatten, \
    Dense, BatchNormalization
from tensorflow_core.python.keras.optimizers import Adam


def get_unet(input_Shape):
    inputs = Input(input_Shape)

    conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
    Drop1 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(Drop1)

    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
    Drop2 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(Drop2)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
    Drop3 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(Drop3)

    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up_conv5, conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up_conv6, conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up_conv7, conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up_conv8, conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)
    #conv10 = Conv2D(2, (1, 1), activation="relu")(conv9)

    flatten = Flatten()(conv9)
    Dense1 = Dense(64, activation='relu')(flatten)
    Dense2 = Dense(6)(Dense1)

    model = Model(inputs, Dense2)

    return model
