
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   

import tensorflow as tf

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import Recall, Precision, MeanIoU, TruePositives, TrueNegatives
from tensorflow.keras import backend as K  

#from model import DeepLabV3Plus_PSPNet_SE
#from model import deeplabv3_plus 
from model import Deeplabv3pMobileViT_S
from tensorflow.keras.layers import Input #For Deeplabv3pMobileVit

from metrics import dice_loss, dice_coef, iou, bce_dice_loss, tversky_loss,focal_tversky_loss

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" shuffle images and masks"""
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

"""load data from our new directory where augmented data is stored- "new_data" folder """
def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y

"""Read Path of images and masks and give numpy array """
def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)    #Image in RGB
    #x = cv2.resize(x, (W, H))    #my images are already 512x512
    x = x/255.0                  #Normalizing pixels in range of 0 to 1
    x = x.astype(np.float32)     #Convert to float and numpy array. 
    return x

def read_mask(path):
    path = path.decode()               #Masks in grayscale -binary 
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   #mask is already normalized as pixels are from 0 to 1.So no need to normalize
    #x = cv2.resize(x, (W, H))    #my masks are already 512x512
    #x = x/255.0                    #mask values 0 and 1 .so no need to normalize
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)              #expanding dimensions (h,w,1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])   #images are RGB
    y.set_shape([H, W, 1])   #masks are grayscale with 1 channel
    return x, y

def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)   
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

if __name__ == "__main__":
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Create Directory - FILES. Directory for storing files - .csv file and model.h5 """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 16
    lr = 1e-4
    num_epochs = 30
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Dataset """
    dataset_path = "new_data"

    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")   
    #test_path = os.path.join(dataset_path, "test")

    train_x, train_y = load_data(train_path)    #load training images and masks
    train_x, train_y = shuffling(train_x, train_y)  #shuffle training images and masks
    valid_x, valid_y = load_data(valid_path)        #load validation images and masks
    #test_x, test_y = load_data(test_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")   #Print length of -training images - training masks
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")   #Print length of -validation images- validation masks
    #print(f"Test : {len(test_x)} - {len(test_y)}") 

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    #test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    #for x,y in train_dataset:
     # print(x.shape, y.shape)

    """ For Deeplabv3pMobileVit"""


    input_tensor = Input(shape=(512, 512, 3), name='image_input')

    model = Deeplabv3pMobileViT_S(input_tensor=input_tensor,
                                    weights=None,
                                    num_classes=2,
                                    OS=16)

    meanIoU = tf.keras.metrics.MeanIoU(num_classes=2)

    model.compile(loss= "binary_crossentropy" , 
                optimizer= Adam(lr), 
                metrics=[dice_coef, iou, Recall(), Precision(), meanIoU])
    #model.summary()
    
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]
    

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

   