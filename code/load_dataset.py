import tensorflow as tf
import hyperparameters as hp
import re
import numpy as np

class TreepediaData(): 
    """Class for containing the training and testing sets"""

    def __init__(self, data_path): 
        self.data_path = data_path

        self.train_data = self.get_data("train")
        self.test_data = self.get_data("test")

    
    def read_filepaths_txt(self, filename): 
        '''
        Reads in dataset
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

    def get_data(self, train_or_test): 
        '''
        Creates Dataset object for testing or training dataset 

        for reference: 
        https://github.com/PacktPublishing/What-s-New-in-TensorFlow-2.0/blob/master/Chapter03/cifar10/cifar10_data_prep.py 
        '''
        path = "data/sample_text_training.txt"

        image_count, images, labels = self.read_filepaths_txt(path)
        print(image_count)

        # images not loaded yet, just filepaths
        list_ds = tf.data.Dataset.from_tensor_slices((images, labels))
        # shuffle 
        list_ds = list_ds.shuffle()

        if train_or_test == "train": 
            # creates batches
            dataset = dataset.shuffle(1000).batch(hp.batch_size)
            dataset = dataset.repeat(hp.num_epochs)
        
        return dataset 

    # def parse_image()

def main():
    dataset_obj = TreepediaData("")

main()