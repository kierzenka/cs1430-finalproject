import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
from PIL import Image
import numpy as np
import glob

import hyperparameters as hp
from models import YourModel, DeepGreenModel
from sequential_models import make_deep_green_seq_model
from load_dataset import TreepediaDataset
from skimage.transform import resize



# model = DeepGreenModel()
# model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))

# # load_checkpoint = os.path.abspath(ARGS.load_checkpoint)
# model.load_weights(os.path.abspath("checkpoints/deep_green_model/050522-165423/your.weights.e068-acc0.0204.h5"), by_name=False)
# model.compile(
#   optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
#   loss=tf.keras.losses.MeanSquaredError(),
#   metrics=["mean_absolute_error"])

def decode_image(img): 
  # Convert the compressed string to a 3D uint8 tensor
  decoded_img = []
  if tf.io.is_jpeg(img):
      decoded_img  = tf.io.decode_jpeg(img, channels=3)
  else: # else image is png
      decoded_img  = tf.io.decode_png(img, channels=3)
      
  # Resize + convert image to float representation
  return tf.image.resize(decoded_img  / 255, [hp.img_height, hp.img_width])

def process_file_line(img_path): 
    # read label in as float
    # load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    # convert image data into numpy array
    img = decode_image(img)
    return img

# os.chdir("/Users/filip/Desktop/CS/CS1430/cs1430-finalproject/code")

model = YourModel()
model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))


model.load_weights(os.path.abspath("checkpoints/your_model/050522-204942/your.weights.e018-acc0.0871.h5"), by_name=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mean_absolute_error"])


cur_chunk = "Pnt_start0_end1000"
filenames = glob.glob('../data/provData/prov_gsv_images/'+ cur_chunk + '/*.jpg')
print(len(filenames))
image_list = tf.stack(list(map(process_file_line, filenames)))
panoIDs = [i.split("/")[-1][:-4] for i in filenames]

output = model.predict(image_list, batch_size=32)

with open("../data/provData/"+cur_chunk+"_labels.txt",'w') as f:
  for i in range(len(output)):
    f.write(panoIDs[i]+","+str(output[i][0])+"\n")
# print(output[:50])
# print(model(tf.expand_dims(image_list[0],axis =0)))

