import tensorflow as tf
import hyperparameters as hp
import re
import numpy as np

class TreepediaData(): 
    """Class for containing the training and testing sets"""

    def __init__(self, data_path): 
        self.data_path = data_path

        self.train_data, self.test_data = self.get_data()

        self.train_data = self.train_data.map(self.process_file_line,
                                              num_parallel_calls=tf.data.AUTOTUNE)
        self.test_data = self.test_data.map(self.process_file_line,
                                        num_parallel_calls=tf.data.AUTOTUNE)

        for image, label in self.train_data.take(1):
            print("Image shape: ", image.numpy().shape)
            print("Label: ", label.numpy())
    
    def read_filepaths_txt(self, filename): 
        '''
        Reads in dataset as list of filepaths
        '''
        file_list = []

        with open(filename, "r") as f:
            for line in f:
                file_list.append(line.strip())

        return len(file_list), tf.constant(file_list)

        # img_list = []
        # label_list = []
        # with open(filename, "r") as f:
        #     for line in f:
        #         img_label_list = re.findall(r'\.\/[\w\_\/ ]+\.jpg', line)
        #         img_list.append(img_label_list[0])
        #         label_list.append(img_label_list[1])

        # return  (len(img_list),
        #          tf.convert_to_tensor(img_list), 
        #          tf.convert_to_tensor(label_list),
        #         ) 

    def get_data(self): 
        '''
        Creates Dataset object for testing or training dataset 

        for reference: 
        https://github.com/PacktPublishing/What-s-New-in-TensorFlow-2.0/blob/master/Chapter03/cifar10/cifar10_data_prep.py 
        '''
        image_count, file_list = self.read_filepaths_txt(self.data_path)

        # images not loaded yet
        list_ds = tf.data.Dataset.from_tensor_slices(file_list)
        # shuffle 
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

        for f in list_ds.take(5):
            print(f.numpy())

        # split the dataset into training and test sets
        val_size = int(image_count * hp.val_split)
        train_ds = list_ds.skip(val_size)
        test_ds = list_ds.take(val_size)
        
        return train_ds, test_ds

    def decode_image(self, img): 
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [hp.img_height, hp.img_width])

    def process_file_line(self, file_line):
        print(file_line)
        print(file_line.numpy())
        img_label_list = re.findall(r'\.\/[\w\_\/ ]+\.jpg', file_line)
        img_path, label_path = img_label_list[0], img_label_list[1]
        # Load the raw data from the file as a string

        label = tf.io.read_file(label_path)
        label = self.decode_image(label)
        
        img = tf.io.read_file(img_path)
        img = self.decode_image(img)
        return img, label

def main():
    dataset_obj = TreepediaData("data/sample_text_training.txt")

main()