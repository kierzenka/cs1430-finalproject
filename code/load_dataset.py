import tensorflow as tf
import hyperparameters as hp
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import data

class GreenDataset(): 
    """Class for containing the training and testing sets"""

    def __init__(self, gsv_data_path, cityscapes_data_path): 
        '''
        Constructor for GreenDataset class
        '''
        # make train & test dataset for gsv data
        gsv_count, gsv_ds = self.make_dataset(gsv_data_path)
        # make train & test dataset for citscapes data
        scapes_count, scapes_ds = self.make_dataset(cityscapes_data_path)
        
        print("gsv, scapes count")
        print(gsv_count)
        print(scapes_count)
        # get total number of images in gsv + scapes datasets
        self.img_count = scapes_count + gsv_count
        # get train + test set for combined dataset
        self.train_data, self.test_data = self.load_dataset(gsv_ds, scapes_ds)
        print(self.train_data.cardinality().numpy())
        print(self.train_data)
    
    def read_filepaths(self, filename): 
        ''' 
        Reads in dataset as list of image filepaths and their labels 

        args:
            filename - path to text file which maps images to their GVI 

        returns: 
            A tuple containing number of images in dataset, a list of GSV or
            Cityscapes images as filepaths, and a list of their corresponding 
            labels as floats.
        '''
        img_list = []
        label_list = []
        # file is 
        with open(filename, "r") as f:
            for line in f:
                line_info = line.split()
                img_list.append(line_info[0])
                label_list.append(float(line_info[1]))
        print("filepath: " + filename + " " + str(len(img_list)))
        return  (len(img_list),
                 tf.convert_to_tensor(img_list), 
                 tf.convert_to_tensor(label_list),
                ) 

    def make_dataset(self, data_path): 
        ''' 
        Creates Dataset object for training the model

        args: 
            data_path - file path to GSV or Cityscapes datasets
        
        returns: 
            The number of images in the dataset and a tensorflow Dataset object
        '''
        # reads in text file containing mapped filepaths
        img_count, images, labels = self.read_filepaths(data_path)
        # images not loaded yet, just file paths
        list_ds = data.Dataset.from_tensor_slices((images, labels))
        # shuffle 
        list_ds = list_ds.shuffle(img_count, reshuffle_each_iteration=False)

        return img_count, list_ds

    def decode_image(self, img): 
        ''' 
        Helper method for process_file_line to load in street level images

        args: 
            img - filepath of image

        returns: 
            A resized color image using float representation
        '''
        # Convert the compressed string to a 3D uint8 tensor
        decoded_img = []
        if tf.io.is_jpeg(img):
            decoded_img  = tf.io.decode_jpeg(img, channels=3)
        else: # else image is png
            decoded_img  = tf.io.decode_png(img, channels=3)
            
        # Resize + convert image to float representation
        return tf.image.resize(decoded_img  / 255, [hp.img_height, hp.img_width])

    def process_file_line(self, img_path, label):
        ''' 
        Processes an element in a Dataset object

        args: 
            img_path - filepath to street level iamge
            label - GVI for that image 
        
        returns: 
            A resized image and its GVI, both in float representation
        '''
        # read label in as float
        label = float(label)
        # load the raw data from the file as a string
        img = tf.io.read_file(img_path)
        # convert image data into numpy array
        img = self.decode_image(img)
        return img, label

    def load_dataset(self, ds1, ds2): 
        ''' 
        Concatenates two datasets together and prepares it to
        be put into the model 
        
        args: 
            ds1 - Dataset object
            ds2 - Dataset object

        returns: 
            Training and testing (validation) Dataset objects
        '''
        # concatenate gsv and cityscapes datasets together
        list_ds = ds1.concatenate(ds2)
        print("load ds list ds")

        # split the dataset into training and test sets
        val_size = int(self.img_count * hp.val_split)
        train_ds = list_ds.skip(val_size)
        test_ds = list_ds.take(val_size)

        # convert to images
        train_ds = train_ds.map(self.process_file_line,
                                num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(self.process_file_line,
                              num_parallel_calls=tf.data.AUTOTUNE)


        images = np.array([x for x, y in train_ds])

        # normalize images using mean and standard deviation
        sample_size = 400
        rand_indices = np.floor(np.random.rand(sample_size)*images.shape[0])
        rand_indices = rand_indices.astype(int)
        sample = images[rand_indices]
        mean = np.sum(sample, axis=0) / sample_size
        stand = np.std(sample,axis=0)
        train_ds = train_ds.map(lambda x,y: ((x-mean)/stand,y))
        test_ds = test_ds.map(lambda x,y: ((x-mean)/stand,y))
        train_ds = train_ds.shuffle(buffer_size=10 * hp.batch_size)
        # repeat and batch
        train_ds = train_ds.batch(hp.batch_size)
        test_ds = test_ds.batch(hp.batch_size)
        # prefetch to increase efficiency
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds
