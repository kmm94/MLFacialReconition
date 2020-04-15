import tensorflow as tf

def Evaluate(y_true, y_pred):
    loss = tf.keras.losses.Huber(y_true, y_pred)
    print("Huber loss: ", loss)
    loss = tf.keras.losses.logcosh(y_true, y_pred)
    print("logcosh loss: ", loss)
    loss = tf.keras.losses.squared_hinge(y_true, y_pred)
    print("squared_hinge loss: ", loss)


