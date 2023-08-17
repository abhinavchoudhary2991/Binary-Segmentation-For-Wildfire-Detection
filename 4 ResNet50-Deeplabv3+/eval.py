#EVALUATION OF MODEL ON TEST IMAGES

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from metrics import dice_loss, dice_coef, iou
from metrics import dice_loss, dice_coef, iou, bce_dice_loss
from metrics import tversky_loss,focal_tversky_loss
from metrics import matthews_correlation, mean_pixel_accuracy

from train import load_data
from tqdm.notebook import tqdm

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    ## image -> ground_truth_mask -> predicted_mask -> overlay predicted_mask*imge
    #Note all the images mentioned above should have same no of dimensions. 
    #H,W will be 512,512. No of channels right now => image:3, mask:1 , y_pred:1, image*y_pred :3 channels
    
    line = np.ones((H, 10, 3)) * 128   #line variable is separator between images. 128 is no of channels.So, LINE IS GRAYSCALE WITH 128 PIXELS.

    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3) Now, mask has also 3 channels
    mask = mask * 255      #pixels are in the range of 0 and 1. So multiply by 255 for visualisation

    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3). PIXELS ARE IN RANGE OF 0 AND 1.
    #y_pred = y_pred * 255    #pixels are in the range of 0 and 1. So multiply by 255 for visualisation


    #Overlay of predicted_mask (y_pred) over original image
    masked_image = image * y_pred
    y_pred = y_pred * 255    #pixels are in the range of 0 and 1. So multiply by 255 for visualisation

    cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)

    #SAVE IMAGES
    cv2.imwrite(save_image_path, cat_images)  



if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Loading model and TRAINED WEIGHTS """
    #with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    #with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss,
               # 'bce_dice_loss': bce_dice_loss}): 

    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss, 
                        'bce_dice_loss': bce_dice_loss, 'tversky_loss': tversky_loss,
                        'focal_tversky_loss': focal_tversky_loss,
        'mean_pixel_accuracy': mean_pixel_accuracy, 'matthews_correlation': matthews_correlation}):

        model = tf.keras.models.load_model("files/model.h5")
        #model.summary()

    """ Load the TEST dataset from "new_data" folder """
    dataset_path = "new_data"

    test_path = os.path.join(dataset_path, "test")
    test_x, test_y = load_data(test_path)
    print(f"Test: {len(test_x)} - {len(test_y)}")


    """ Evaluation and Prediction """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):

        """ Extract the names of images without extension-.png """

        name = x.split("/")[-1].split(".")[0]   #For colab
        
        #print(name)
        #break

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0                           #print(x.shape): (512,512,3) so, no need to resize
        x = np.expand_dims(x, axis=0)             #print(x.shape): (1, 512,512,3)
        
        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)   #No need of normalizing as mask is already normalized
        #_, mask = cv2.threshold(cv2.imread(y, cv2.IMREAD_GRAYSCALE), 127, 1, cv2.THRESH_BINARY)  #FOR CORSICAN, MASK is 0 and 255. SO RUN THIS FOR CORSICAN

        # PREDICTION 
        y_pred = model.predict(x)[0]     #taking 1st element of the list of all images
        y_pred = np.squeeze(y_pred, axis=-1)   #Shape is 512x512x1. Now it will be 512x512
        y_pred = y_pred > 0.5                     #APPLY binary THRESHOLDING. Value>0.5 will become 1, rest will remain 0.
        y_pred = y_pred.astype(np.int32)

        #Saving the prediction 
        
        save_image_path = f"results/{name}.png"
        save_results(image, mask, y_pred, save_image_path)  #from function save_results

        
        #FOR CALCULATIONS

        #Flatten the array"""
        mask = mask.flatten()
        mask = mask.astype(np.int32)    #NOT INCLUDED IN ORIGINAL 
        y_pred = y_pred.flatten()

        print("Mask dtype:", mask.dtype)
        print("y_pred dtype:", y_pred.dtype)

       

        # Calculating the metrics values 

        acc_value = accuracy_score(mask, y_pred)
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary")
        mean_pixel_acc_value = mean_pixel_accuracy(mask, y_pred)
        mcc_value = matthews_correlation(mask, y_pred)
        
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value,
                  mean_pixel_acc_value, mcc_value])


    #Metrics values 
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")
    print(f"Mean Pixel Accuracy: {score[5]:0.5f}")  # Add this line
    print(f"Matthews Correlation Coefficient: {score[6]:0.5f}")  # Add this line

    # SAVE METRICS USING PANDAS INTO .CSV FILE

    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision",
                                 "Mean Pixel Accuracy", "Matthews Correlation"])
    df.to_csv("files/score.csv")


