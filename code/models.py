import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer =tf.keras.optimizers.RMSprop(learning_rate=hp.learning_rate, momentum=hp.momentum)

        self.resnet_preprocess = tf.keras.applications.resnet.preprocess_input
        self.resnet = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

        # From the research paper: https://arxiv.org/pdf/1808.04754.pdf 
        # Using a 50 layered ResNet as the
        # base architecture, we add 3 more layers of dense connections
        # at the end, with a final layer consisting of a single sigmoid
        # unit.
        self.classification_head = [
            Flatten(),
            Dense(32,activation='relu'),
            Dense(32,activation='relu'),
            Dense(32,activation='relu'),
            Dense(1,activation='sigmoid')
        ]

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
       # Using mean squared error will maximize the Pearson's correlation coefficient
       # https://stats.stackexchange.com/questions/301659/mse-as-a-proxy-to-pearsons-correlation-in-regression-problems 
       mse = tf.keras.losses.MeanSquaredError()
       return mse(labels, predictions)