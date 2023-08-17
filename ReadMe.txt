BRIEF: The codes are keras implementation of binary semantic segmentation using 4 different architectures - MobileViT_S, MobileViT_XS, Mobilenetv3 and ResNet-50. THese architectures are used as encoders and integrated with Deeplabv3+ segmentation framework, respectively.

Steps to run the codes:- 

1. Create a folder where your dataset is stored. The images should be in "images" subfolder and masks should be in "masks" subfolder. 

2. Run data.py

data.py: This script will split the images and masks in the proportion of 70% training, 15% testing and 15% validation. It will create a new folder named "new_data" with subfolders "train", "valid" and "test". Each of the subfolders will have two more subfolders "image" where images will get stored and "mask" where masks will get stored. To the images and masks in "train" folder, it will further apply augmentation.


3. Folders MobileViT_S-Deeplabv3+, MobileViT_XS-Deeplabv3+, Mobilenetv3-Deeplabv3+ and Resnet50-Deeplabv3+ has model architectures. To use this, rename the architectures containing the main model architecture as "model.py" and keep it in same folder as other scripts.

metrics.py : It has all metrics and loss functions

layers1.py : This script has functions which are common for all models. It has 			     structures for ASPP module and Decoder Module of Deeplabv3+.  Hence, this 		     script should be there with scripts having corresponding model 			     architectures. (Not required for ResNet-50)

4. Run model.py to see model architecture

TRAINING
5. Run train.py to begin training.

train.py : It has hyperparameters. Choose suitable loss function in model.compile(). It will create a folder "files" where model weights are stored as "model.h5", training progress is stored as "data.csv" file.

EVALUATION
6. Run eval.py to evaluate the model on test images

eval.py : It will evaluate the model on test images and store the results as .png images in "results" folder. It will also store the results in "files" folder as a "score.csv" file.


7. PREDICTION
Create a new folder. Name it "test_images". Create two sub-folders "image" and "mask". Put the unseen images to be predicted inside "image" folder.

Run script predict.py

The predicted masks can be seen in "mask" folder.


### TRAINING RESULTS###
This folder contain the training results of all datasets. The 
loss functions that gave the best results for each architecture- "data1.csv" for MobileViT_S, "data2.csv" for MobileViT_XS, "data3.csv" for Mobilenetv3 and data4.csv for ResNet50 model

 