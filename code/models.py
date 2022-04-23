import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer =tf.keras.optimizers.RMSprop(learning_rate=hp.learning_rate, momentum=hp.momentum)
       #  self.architecture = self.architecture = [Conv2D(64, 3, padding='same', activation='relu'),
       #                                              Conv2D(128, 3, padding='same', activation='relu'),
       #                                              MaxPool2D(),
       #                                              Conv2D(200,5,padding='same',strides=(2,2),activation='relu'),
       #                                              Conv2D(250,5,padding='same',strides=(2,2),activation='relu'),
       #                                              MaxPool2D(strides=(2,2)),
       #                                              Dropout(0.1),
       #                                              Flatten(),
       #                                              #Dense(500, activation='relu'),
       #                                              Dense(100,activation='relu'),
       #                                              Dense(hp.num_classes,activation='softmax')]

        self.resnet_preprocess = tf.keras.applications.resnet.preprocess_input
        self.resnet = tf.keras.applications.resnet50.ResNet5(weights='imagenet')
        self.classification_head = [MaxPool2D(strides=(2,2)),Flatten(),Dense(200,activation='relu'),Dense(100,activation='relu'),Dense(1,activation='sigmoid')]

        for l in self.resnet.layers:
               l.trainable = False

    def call(self, x):
        """ Passes input image through the network. """
        x = self.resnet_preprocess(x)
        x = self.resnet(x)
        for layer in self.classification_head:
            x = layer(x)
       #      print(x.shape)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
       """ Loss function for the model. """
       # TODO: Select a loss function for your network (see the documentation
       #       for tf.keras.losses)
       # lf = tf.keras.losses.SparseCategoricalCrossentropy()
       # return lf(labels, predictions, from_logits=False)
       return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
       # pass