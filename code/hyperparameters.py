"""
Number of epochs. If you experiment with more complex networks you
might need to increase this. Likewise if you add regularization that
slows training.
"""
num_epochs = 150

"""
A critical parameter that can dramatically affect whether training
succeeds or fails. The value for this depends significantly on which
optimizer is used. Refer to the default learning rate parameter
"""
learning_rate = 1e-4
"""
Momentum on the gradient (if you use a momentum-based optimizer)
"""
momentum = 0.01

"""
Resize image size for task 1. Task 3 must have an image size of 224,
so that is hard-coded elsewhere.
"""
img_size = 224

"""
Image height and width for resizinn
"""
img_height = 224
img_width = 224

"""
Sample size for calculating the mean and standard deviation of the
training data. This many images will be randomly seleted to be read
into memory temporarily.
"""
preprocess_sample_size = 400

"""
Maximum number of weight files to save to checkpoint directory. If
set to a number <= 0, then all weight files of every epoch will be
saved. Otherwise, only the weights with highest accuracy will be saved.
"""
max_num_weights = 5

"""
Defines the number of training examples per batch.
You don't need to modify this.
"""
batch_size = 50

"""
The number of image scene classes. Don't change this.
"""
num_classes = 15

"""
A float between 0 and 1 the represents the fraction of data to 
reserve for validation.
"""
val_split = 0.2
