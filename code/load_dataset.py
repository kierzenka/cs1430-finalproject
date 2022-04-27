import tensorflow as tf
import hyperparameters as hp
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class TreepediaData(): 
    """Class for containing the training and testing sets"""

    def __init__(self, data_path): 
        self.data_path = data_path

        self.train_data, self.test_data = self.get_data()

        self.train_data = self.train_data.map(self.process_file_line,
                                              num_parallel_calls=tf.data.AUTOTUNE)
        self.test_data = self.test_data.map(self.process_file_line,
                                        num_parallel_calls=tf.data.AUTOTUNE)

        plt.figure(figsize=(10, 10))
        for image, label in self.train_data.take(1):
            plt.imsave("test.jpg", label.numpy().astype("uint8"))
            plt.title("label")
            plt.axis("off")
    
    def read_filepaths_txt(self, filename): 
        '''
        Reads in dataset as list of filepaths
        '''
        img_list = []
        label_list = []
        with open(filename, "r") as f:
            for line in f:
                img_label_list = re.findall(r'\.\/[\w\_\/ ]+\.jpg', line)
                img_list.append(img_label_list[0])
                label_list.append(img_label_list[1])

        return  (len(img_list),
                 tf.convert_to_tensor(img_list), 
                 tf.convert_to_tensor(label_list),
                ) 

    def get_data(self): 
        '''
        Creates Dataset object for testing or training dataset 

        for reference: 
        https://github.com/PacktPublishing/What-s-New-in-TensorFlow-2.0/blob/master/Chapter03/cifar10/cifar10_data_prep.py 
        '''
        image_count, images, labels = self.read_filepaths_txt(self.data_path)

        test_image = tf.io.read_file(images[0])
        test_image = tf.io.decode_jpeg(test_image, channels=3)
        print(test_image.numpy().shape)


        # images not loaded yet, just file paths
        list_ds = tf.data.Dataset.from_tensor_slices((images, labels))
        # shuffle 
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

        # split the dataset into training and test sets
        val_size = int(image_count * hp.val_split)
        train_ds = list_ds.skip(val_size)
        test_ds = list_ds.take(val_size)
        
        return train_ds, test_ds

    def decode_image(self, img, grayscale): 
        # Convert the compressed string to a 3D uint8 tensor
        # img = tf.io.decode_jpeg(img, channels=3)
        if grayscale: 
            img = tf.io.decode_jpeg(img, channels=1)
        else: 
            img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [hp.img_height, hp.img_width])

    def process_file_line(self, img_path, label_path):
        # Load the raw data from the file as a string
        label = tf.io.read_file(label_path)
        # TODO: double check how to hand grayscale
        label = self.decode_image(label, grayscale=False)
        
        img = tf.io.read_file(img_path)
        img = self.decode_image(img, grayscale=False)
        return img, label

def main():
    dataset_obj = TreepediaData("data/sample_text_training.txt")

main()