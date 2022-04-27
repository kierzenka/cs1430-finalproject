from sklearn.feature_extraction import img_to_graph
import tensorflow as tf
import hyperparameters as hp
import re

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

        return (tf.constant(img_list), tf.constant(label_list))

    def get_data(self, train_or_test): 
        path = "data/sample_text_training.txt"
        dataset = tf.data.Dataset.from_tensor_slices(self.read_filepaths_txt(path))
        print(dataset)
        return dataset 

def main():
    dataset_obj = TreepediaData("")

main()