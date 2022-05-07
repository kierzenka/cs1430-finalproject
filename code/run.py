import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel, DeepGreenModel
from sequential_models import make_deep_green_seq_model
from load_dataset import TreepediaDataset
from skimage.transform import resize
# from tensorboard_utils import \
#         ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver
from tensorboard_utils import CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.cm as cm
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """
    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-gsv',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the gsv dataset is stored.')
    parser.add_argument(
        '--data-cityscapes',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the cityscapes dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--lime-image',
        default=None,
        help='''Name of an image in the dataset to use for LIME evaluation.''')
    parser.add_argument(
        '--gradcam-image',
        default=None,
        help='''Name of an image in the dataset to use for GRAD-cam evaluaton''')
    parser.add_argument(
        '--deep-green',
        default=None,
        help='''Trains using the Deep Green Diagnostics model''')
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='''Uses Sequential model (necessary for GRAD-cam)''')

    return parser.parse_args()

def LIME_explainer(model, path):
    """
    This function takes in a trained model and a path to an image and outputs 5
    visual explanations using the LIME model
    """

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True, path=None):
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        data = mark_boundaries(temp/2+0.5,mask)
        print(np.min(data))
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        plt.imsave(fname=path, arr=data)


    # Read the image and preprocess it as before
    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3))
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True, path="top5superpixels.png")

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False, path="top5withrestofimage.png")

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False, path="prosandcons.png")

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imsave(fname="mapweighttosuperpixel.png", arr=heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())

def superimposed_gradcam(img_path, heatmap, cam_path="superimposed.png", alpha=0.6):
    # Load the original image
    img = Image.open(img_path).convert("RGBA").resize((244, 244))

    # Use jet colormap to colorize heatmap
    jet = cm.jet
    jet_heatmap = jet(heatmap)
    jet_heatmap = np.uint8(255 * jet_heatmap)

    # Convert jet heatmap to PIL image
    jet_heatmap = Image.fromarray(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((244, 244))
    jet_heatmap = jet_heatmap.convert("RGBA")

    # Superimpose the heatmap on original image
    # superimposed = Image.blend(img, jet_heatmap, alpha)

    # Save the superimposed image
    # superimposed.save(cam_path)
    # jet_heatmap.save("test.png")

    # create mask for superimposing
    # paste_mask = jet_heatmap.split()[3].point(lambda i: i * 0.2)

    # superimpose images
    # img.paste(jet_heatmap, (0,0), mask=paste_mask)
    # img.save(cam_path)
    img.save("test.png")

def make_gradcam_heatmap(img_path, model, last_conv_layer_name, pred_index=None):
    
    def get_img_array(img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(244, 244))
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = tf.keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        return array

    model.layers[-1].activation = None

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(get_img_array(img_path))
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    print(heatmap.min())
    print(heatmap.max())

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap =  (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # plt.matshow(heatmap)
    # plt.show()
    print(heatmap.min())
    print(heatmap.max())
    plt.imsave("heatmap.png", heatmap)
    return heatmap

def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        # ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]

    # Begin training
    print("datasets size")
    print(datasets.train_data.cardinality().numpy())
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data_gsv):
        ARGS.data_gsv = os.path.abspath(ARGS.data_gsv)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = TreepediaDataset(ARGS.data_gsv, ARGS.data_cityscapes)

   
    model, checkpoint_path, logs_path = None, None, None
    if ARGS.deep_green:
        if ARGS.sequential:
            model = make_deep_green_seq_model(hp.img_size, hp.img_size)
            checkpoint_path = "checkpoints" + os.sep + \
                "deep_green_model_seq" + os.sep + timestamp + os.sep
            logs_path = "logs" + os.sep + "deep_green_model_seq" + \
                os.sep + timestamp + os.sep
        else:
            model = DeepGreenModel()
            model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
            checkpoint_path = "checkpoints" + os.sep + \
                "deep_green_model" + os.sep + timestamp + os.sep
            logs_path = "logs" + os.sep + "deep_green_model" + \
                os.sep + timestamp + os.sep
            
    else:
        model = YourModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
             os.sep + timestamp + os.sep
        

    # Print summary of model
    model.summary()

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph

    if ARGS.sequential and ARGS.deep_green:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["mean_absolute_error"]
        )
    else:
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss_fn,
            metrics=["mean_absolute_error"])

    if ARGS.evaluate:
        test(model, datasets.test_data)

        # TODO: change the image path to be the image of your choice by changing
        # the lime-image flag when calling run.py to investigate
        # i.e. python run.py --evaluate --lime-image test/Bedroom/image_003.jpg
        path = ARGS.data_gsv + os.sep + ARGS.lime_image
        LIME_explainer(model, path, datasets.preprocess_fn)
    elif ARGS.lime_image or ARGS.gradcam_image:
        if ARGS.lime_image:
            lime_path = ARGS.lime_image
            LIME_explainer(model, lime_path)
        if ARGS.gradcam_image: 
            gradcam_path = ARGS.gradcam_image
            # not sure if this will work
            last_conv_layer = "resnet50"
            if ARGS.deep_green: 
                last_conv_layer = "conv5"
            heatmap = make_gradcam_heatmap(gradcam_path, model, last_conv_layer)
            superimposed_gradcam(gradcam_path, heatmap)
            
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)

# Make arguments global
ARGS = parse_args()

main()