import tensorflow as tf
import hyperparameters as hp

class TreepediaData(): 
    """Class for containing the training and testing sets"""

    def __init__(self, data_path): 
        self.data_path = data_path

        self.train_data = self.get_data("train")
        self.test_data = self.get_data("test")

    def get_data(self, train_or_test): 
        dataset = tf.data.TextLineDataset("/data/sample_text_training.txt")
        for line in dataset.take(5): 
            print(line)
        return dataset 

def main():
    dataset_obj = TreepediaData("")

main()