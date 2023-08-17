import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm   #tqdm is to check the progress 
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate
from albumentations import HorizontalFlip, CoarseDropout, RandomBrightness, RandomContrast

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split =0.3):

    """ Loading the images and masks """
    X = sorted(glob(os.path.join(path, "images", "*")))       #Original images have .jpg extension
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))    #Original Masks have .png extension
    
    """
    #TO VERIFY IMAGES AND MASKS ARE IN SAME ORDER
    
    for x,y in zip(X,Y):     #It means for all x,y in Images,Masks
        print(x,y)

        x= cv2.imread(x)
        cv2.imwrite("x.jpg",x)

        y= cv2.imread(y)
        cv2.imwrite("Y.png",y*255)    #If mask is binary 0/1 >>cv2.imwrite("Y.png",y*255) 
        break 

if __name__ == "__main__":

    # Seeding 
    np.random.seed(42)

    # Load the dataset 
    data_path = "flame_dataset"
    load_data(data_path)
    """

    split_size = int(len(X) * split)    

    """ Splitting the training data into training and validation """
    train_x, temp_x = train_test_split(X, test_size= split_size, random_state=42)
    train_y, temp_y = train_test_split(Y, test_size=split_size, random_state=42)

    """ Splitting the data into training and testing """
    valid_x, test_x = train_test_split(temp_x, test_size= 0.5, random_state=42)
    valid_y, test_y = train_test_split(temp_y, test_size= 0.5, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

    """
    #Spliting the data into training and testing 

    
    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y)
    """

def augment_data(images, masks, save_path, augment=True):
   
    H = 512
    W = 512

    for x, y in tqdm(zip(images, masks), total=len(images)):

        # Extract the name 
        #name = x.split("\\")[-1].split(".")[0]    #print and check that you get name of image without extension (without .jpg)
        name = x.split("/")[-1].split(".")[0]      #For colab. Works for Hilda as well
        #print(name)
        #break

        # Reading the image and mask 
        x = cv2.imread(x, cv2.IMREAD_COLOR)   #read images
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)   #read masks

        # DATA  Augmentation 
        if augment == True:

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            #x2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            #y2 = y

            #aug = ChannelShuffle(p=1)
            #augmented = aug(image=x, mask=y)
            #x3 = augmented['image']
            #y3 = augmented['mask']

            #aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
            #augmented = aug(image=x, mask=y)
            #x4 = augmented['image']
            #y4 = augmented['mask']

            aug = RandomRotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            aug = RandomBrightness(p=1)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            aug = RandomContrast(p=1)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']



            X = [x, x1,  x5, x6, x7]
            Y = [y, y1,  y5, y6, y7]

        else:
            X = [x]   #image list stored in X
            Y = [y]    # mask list stored in Y

        index = 0
        for i, m in zip(X, Y):   

            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1 


if __name__ == "__main__":
    # Seeding 
    np.random.seed(42)

    # Load the dataset 
    #data_path = "data_corsican"
    data_path = "combined_corsican_flame"    #PATH OF DATASET FOLDER

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_path)

    print(f"Train:\t {len(train_x)} - {len(train_y)}")
    print(f"Valid:\t {len(valid_x)} - {len(valid_y)}")
    print(f"Test:\t {len(test_x)} - {len(test_y)}")

    # Create directories to save the augmented data 
    #NOTE: Data augmentation is done only on training data and not on test data.

    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/valid/image/")
    create_dir("new_data/valid/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    # Data augmentation 
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(valid_x, valid_y, "new_data/valid/", augment=False)
    augment_data(test_x, test_y, "new_data/test/", augment=False)

