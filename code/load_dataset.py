import tensorflow as tf
import hyperparameters as hp
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import data

class TreepediaDataset(): 
    """Class for containing the training and testing sets"""

    def __init__(self, gsv_data_path, cityscapes_data_path): 
        # make train & test dataset for gsv data
        gsv_count, gsv_ds = self.make_dataset(gsv_data_path)
        # make train & test dataset for citscapes data
        scapes_count, scapes_ds = self.make_dataset(cityscapes_data_path)
        
        # get total number of images in gsv + scapes datasets
        self.img_count = scapes_count + gsv_count
        # get train + test set for combined dataset
        self.train_data, self.test_data = self.load_dataset(gsv_ds, scapes_ds)

        # plt.figure(figsize=(10, 10))
        #for image, label in self.train_data.take(1):
            #plt.imsave("image_test.jpg", image.numpy())
            #plt.axis("off")
            #plt.imsave("label_test.jpg", label.numpy())
    
    def read_filepaths(self, filename): 
        '''
        Reads in dataset as list of filepaths
        '''
        img_list = []
        label_list = []
        with open(filename, "r") as f:
            for line in f:
                line_info = line.split()
                img_list.append(line_info[0])
                label_list.append(float(line_info[1]))

                # img_label_list = re.findall(r'\.\/[\w\_\/ ]+\.jpg', line)
                # img_list.append(img_label_list[0])
                # label_list.append(img_label_list[1])

        return  (len(img_list),
                 tf.convert_to_tensor(img_list), 
                 tf.convert_to_tensor(label_list),
                ) 

    def make_dataset(self, data_path): 
        '''
        Creates Dataset object for testing or training dataset 

        for reference: 
        https://github.com/PacktPublishing/What-s-New-in-TensorFlow-2.0/blob/master/Chapter03/cifar10/cifar10_data_prep.py 
        '''
        # reads in text file containing mapped filepaths
        img_count, images, labels = self.read_filepaths(data_path)
        # images not loaded yet, just file paths
        list_ds = data.Dataset.from_tensor_slices((images, labels))
        # shuffle 
        list_ds = list_ds.shuffle(img_count, reshuffle_each_iteration=False)

        return img_count, list_ds

    def decode_image(self, img): 
        # Convert the compressed string to a 3D uint8 tensor
        decoded_img = []
        if tf.io.is_jpeg(img):
            decoded_img  = tf.io.decode_jpeg(img, channels=3)
        else: # else image is png
            decoded_img  = tf.io.decode_png(img, channels=3)
            
        # Resize + convert image to float representation
        return tf.image.resize(decoded_img  / 255, [hp.img_height, hp.img_width])

    def process_file_line(self, img_path, label_path): 
        # read label in as float
        label = float(label_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(img_path)
        # convert image data into numpy array
        img = self.decode_image(img)
        return img, label

    def load_dataset(self, ds1, ds2): 
        # concatenate gsv and cityscapes datasets together
        list_ds = ds1.concatenate(ds2)
       
        # TODO: check sets are splitting properly
        # split the dataset into training and test sets
        val_size = int(self.img_count * hp.val_split)
        train_ds = list_ds.skip(val_size)
        test_ds = list_ds.take(val_size)

        # convert to images
        train_ds = train_ds.map(self.process_file_line,
                                num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(self.process_file_line,
                              num_parallel_calls=tf.data.AUTOTUNE)
        # shuffle data again
        # TODO: change batch size?
        train_ds = train_ds.shuffle(buffer_size=10 * hp.batch_size)
        # repeat and batch
        train_ds = train_ds.repeat(hp.num_epochs).batch(hp.batch_size)
        # prefetch to increase efficiency
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)