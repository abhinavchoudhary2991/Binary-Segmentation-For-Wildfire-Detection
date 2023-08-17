#PREDICTION OF MODEL ON UNKNOWN IMAGES


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf

from tensorflow.keras.utils import CustomObjectScope

#from metrics import dice_loss, dice_coef, iou
from metrics import dice_loss, dice_coef, iou, bce_dice_loss
from metrics import tversky_loss,focal_tversky_loss
from metrics import matthews_correlation, mean_pixel_accuracy
from train import create_dir
from PIL import Image

""" Global parameters """
H = 512
W = 512

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files #PREDICTED MASK OF UNKNOWN IMAGES WILL BE STORED HERE"""
    create_dir("test_images/mask")

    """ Loading TRAINED model WEIGHTS """
    #with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss, 
                        'bce_dice_loss': bce_dice_loss, 'tversky_loss': tversky_loss,
                        'focal_tversky_loss': focal_tversky_loss,
                         'mean_pixel_accuracy': mean_pixel_accuracy, 
                         'matthews_correlation': matthews_correlation}):

        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset #LOAD UNKNOWN IMAGES WHOSE MASK IS TO BE PREDICTED"""
    data_x = glob("test_images/image/*")

    for path in tqdm(data_x, total=len(data_x)):
        """ Extracting name """
        #name = path.split("\\")[-1].split(".")[0]
        name = path.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))   #RESIZE UNKNOWN IMAGE TO 512X512
        x = x/255.0                     #NORMALIZE 
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction OF MASK WITHOUT ANY THRESHOLDING """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))   #RESISZE MASK TO ORIGINAL UNKNOWN IMAGE SIZE AND NOT 512X512
        
        #y = y > 0.7    #############################NOT IN ORIGINAL
        #y = y.astype(np.uint8) ####################NOT IN ORIGINAL
        
        y = np.expand_dims(y, axis=-1) #value of y is in the range of 0 to 1 and not 0/1.

        """ Save the image """
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128

        cat_images = np.concatenate([image, line, masked_image], axis=1)
        
        cv2.imwrite(f"test_images/mask/{name}.png", cat_images)




