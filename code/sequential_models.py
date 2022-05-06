import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Flatten, Dense, SpatialDropout2D, Dropout

def make_deep_green_seq_model(img_height, img_width): 
    model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(img_height, img_width, 3)),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same',
            strides=(1, 1), activation='relu'), 
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(filters=128, kernel_size=(5, 5), padding='same',
            strides=(1, 1), activation='relu'), 
        MaxPool2D(pool_size=(4, 4)),

        Conv2D(filters=256, kernel_size=(7, 7), padding='same',
            strides=(1, 1), activation='relu'), 
        Conv2D(filters=512, kernel_size=(7, 7), padding='same',
            strides=(1, 1), activation='relu'), 
        MaxPool2D(pool_size=(6, 6)),
        SpatialDropout2D(0.5),

        Flatten(),
        Dense(units=1024, activation="relu"),
        Dense(units=512, activation="relu"), 
        Dropout(rate=0.5),
        Dense(units=256, activation="relu"), 
        Dense(units=128, activation="relu"), 
        Dense(units=64, activation="relu"),
        Dense(units=1, activation="sigmoid")
    ])
    
    return model