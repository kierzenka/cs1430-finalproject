import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Flatten, Dense, SpatialDropout2D, Dropout

def make_deep_green_seq_model(img_height, img_width): 
    model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(img_height, img_width, 3)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', strides=1, padding='same'),
        Conv2D(32, kernel_size=(3, 3), activation='relu', strides=1, padding='same'),
        MaxPool2D((2,2), strides=(2,2)), 
        Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1, padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1, padding='same'),
        MaxPool2D((2,2), strides=(2,2)), 
        Conv2D(128, kernel_size=(5, 5), activation='relu', strides=1, padding='same'),
        Conv2D(128, kernel_size=(5, 5), activation='relu', strides=1, padding='same'),
        MaxPool2D((2,2), strides=(2,2)), 
        Conv2D(256, kernel_size=(7, 7), activation='relu', strides=1, padding='same'),
        Conv2D(512, kernel_size=(7, 7), activation='relu', strides=1, padding='same'),
        MaxPool2D((2,2), strides=(2,2)), 
        SpatialDropout2D(0.5),
        Flatten(),
        Dropout(rate=0.5),
        Dense(256, activation='relu'), # reduced the size because of training time
        Dense(128, activation='relu'), 
        Dense(64, activation='relu'), 
        Dense(1, activation='sigmoid')
    ])
    
    return model