import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

import hyperparameters as hp

class Datasets(): 
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """
    def __init__(self, data_path):

        self.data_path = data_path

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # Mean and std for standardization
        self.mean = np.zeros((hp.img_size,hp.img_size,3))
        self.std = np.ones((hp.img_size,hp.img_size,3))
        self.calc_mean_and_std()

        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), False, False)

    def get_data(self, data_path, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/"
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:
            # TODO: if we have time we can figure this out
            pass 
        else:
            train_data = tf.keras.utils.image_dataset_from_directory(
                data_path, 
                labels=None,
                color_mode="rgb",
                batch_size=hp.batch_size,
                image_size=(hp.img_size, hp.image_size),
                #TODO: what should we set seed to?
                seed=1,
                validation_split=hp.validation_split,
                subset="training",
                #TODO: decide whether we should set below to True
                crop_to_aspect_ratio=False
            )

            test_data = tf.keras.utils.image_dataset_from_directory(
                data_path, 
                labels=None,
                color_mode="rgb",
                batch_size=hp.batch_size,
                image_size=(hp.img_size, hp.image_size),
                #TODO: what should we set seed to?
                seed=1 ,
                validation_split=hp.validation_split,
                subset="validation",
                #TODO: decide whether we should set beloe to True
                crop_to_aspect_ratio=False
            )


        return data_gen

        