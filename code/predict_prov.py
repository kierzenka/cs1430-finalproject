import tensorflow as tf
import numpy as np
import glob
import hyperparameters as hp
from models import DCNNModel

def decode_image(img): 
    ''' 
    Convert the compressed string to a 3D float tensor

    args: 
        img - filepath to image to decompress

    returns: 
        3D float tensor representing a resized image
    '''
    decoded_img = []
    if tf.io.is_jpeg(img):
        decoded_img  = tf.io.decode_jpeg(img, channels=3)
    else: # else image is png
        decoded_img  = tf.io.decode_png(img, channels=3)
      
    # Resize + convert image to to float representation
    return tf.image.resize(decoded_img  / 255, [hp.img_height, hp.img_width])

def process_file_line(img_path): 
    ''' 
    Processes an element in a Dataset object

    args: 
        img_path - filepath to street level iamge
        
    returns: 
        A resized image in float representation
    '''
    # read label in as float
    # load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    # convert image data into numpy array
    img = decode_image(img)
    return img

# initialize DCNN model
model = DCNNModel()
model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))


# Load trained weights for DCNN model
model.load_weights("/Users/alexkamper/Desktop/cs1430-finalproject/code/checkpoints/your_model/050822-201054/your.weights.e013-acc0.0888.h5",by_name=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mean_absolute_error"])


cur_chunk = "Pnt_start1000_end2000"
# filenames = glob.glob('../data/provData/prov_gsv_images/'+ cur_chunk + '/*.jpg')
filenames = glob.glob("/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/provImages/"+cur_chunk+"/*.jpg")
print(len(filenames))
image_list = tf.stack(list(map(process_file_line, filenames)))
panoIDs = [i.split("/")[-1][:-4] for i in filenames]

sample_size = 400
rand_indices = np.floor(np.random.rand(sample_size)*image_list.shape[0])
                #print(rand_indices[0:10])
                        #print(rand_indices.dtype)
rand_indices = rand_indices.astype(int)
sample = image_list.numpy()[rand_indices]
mean = np.sum(sample, axis=0) / sample_size
stand = np.std(sample,axis=0)

image_list = tf.map_fn(fn=lambda x: (x-mean)/stand, elems=image_list)


output = model.predict(image_list, batch_size=32)

with open("/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/"+cur_chunk+"_labels.txt",'w') as f:
  for i in range(len(output)):
    f.write(panoIDs[i]+","+str(output[i][0])+"\n")