
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)
    

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


# Dice_loss = 1- dice_coefficient

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)



##### ADDING NEW LOSS FUNCTIONS ####

def bce_dice_loss(y_true, y_pred):
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss_value = dice_loss(y_true, y_pred)
    return bce_loss + dice_loss_value


def weighted_bce_dice_loss(y_true, y_pred, w1=0.3, w2=0.7):
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss_value = dice_loss(y_true, y_pred)
    return w1 * bce_loss + w2 * dice_loss_value



def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)

    return 1 - ((true_pos + smooth) / (true_pos + alpha*false_neg + beta*false_pos + smooth))


def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=2.0):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)

    tversky = (true_pos + smooth) / (true_pos + alpha*false_neg + beta*false_pos + smooth)
    return K.pow((1 - tversky), gamma)


import keras.backend as K

def matthews_correlation(y_true, y_pred):
    #Compute the Matthews correlation coefficient
    y_true = K.cast(K.flatten(y_true), dtype=tf.float32)
    y_pred = K.cast(K.flatten(y_pred), dtype=tf.float32)

    true_positives = K.sum(y_true * y_pred)
    true_negatives = K.sum((1 - y_true) * (1 - y_pred))
    false_positives = K.sum((1 - y_true) * y_pred)
    false_negatives = K.sum(y_true * (1 - y_pred))

    numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    denominator = K.sqrt((true_positives + false_positives) *
                        (true_positives + false_negatives) *
                        (true_negatives + false_positives) *
                        (true_negatives + false_negatives) + K.epsilon())

    mcc = numerator / denominator
    return mcc


import tensorflow as tf

def mean_pixel_accuracy(y_true, y_pred):
    total_pixels = tf.cast(tf.size(y_true), dtype=tf.float32)
    correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))
    accuracy = correct_pixels / total_pixels
    return accuracy


