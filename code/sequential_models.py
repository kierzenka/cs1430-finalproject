import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Flatten, Dense

def make_deep_green_seq_model(img_height, img_width): 
    model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(244, 244, 3)),
        Conv2D(32, kernel_size=(5, 5), activation='relu', strides=1, padding='same', name='conv1'),
        MaxPool2D((2,2), strides=(2,2)), 
        Conv2D(64, kernel_size=(5, 5), activation='relu', strides=1, padding='same', name='conv2'),
        MaxPool2D((2,2), strides=(2,2)), 
        Conv2D(128, kernel_size=(5, 5), activation='relu', strides=1, padding='same', name='conv3'),
        MaxPool2D((2,2), strides=(2,2)), 
        Conv2D(256, kernel_size=(5, 5), activation='relu', strides=1, padding='same', name='conv4'),
        MaxPool2D((2,2), strides=(2,2)), 
        Conv2D(512, kernel_size=(5, 5), activation='relu', strides=1, padding='same', name='conv5'),
        MaxPool2D((2,2), strides=(2,2)), 
        Flatten(),
        Dense(32, activation='relu'), # reduced the size because of training time
        Dense(32, activation='relu'), 
        Dense(32, activation='relu'), 
        Dense(1, activation='sigmoid')
    ])
    
    return model