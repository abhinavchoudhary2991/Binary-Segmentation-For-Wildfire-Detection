#ResNet-50 deeplabv3+


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

"""The architecture of deeplabv3+ is modified with the addition of a Squeeze-and-Excitation 
block (SE block) that adaptively recalibrates channel-wise feature responses.
This function defines a Squeeze-and-Excitation block, which first applies GlobalAveragePooling2D 
to generate channel-wise statistics. Then, these statistics go through two Dense layers 
to produce weights for each channel. The output feature maps of this block are rescaled 
by these weights."""
def SqueezeAndExcite(inputs, ratio=8):
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = init * se
    return x


"""ASPP: Atrous Spatial Pyramid Pooling (ASPP) is used to capture multi-scale information. 
The function applies different dilation rates (6, 12, and 18) in three parallel branches, 
captures image-level features in another branch, and uses 1x1 convolution in the final branch.
 The output of these branches is then concatenated."""

def ASPP(inputs):

    #print(inputs.shape)  #Output :  (None,32,32,1024). width =32, height=32, no of o/p channels = 1024

    """ 1. Image Pooling """
    shape = inputs.shape
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)  #shape[1] =32 , shape[2] =32,  print(y1.shape) = (None,1,1,1024)
    y1 = Conv2D(256, 1, padding="same", use_bias=False)(y1)        # no of features =256, kernel = 1x1
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)   #bilinear upsampling. 
    #print(y1.shape)                                #(none, 32, 32, 256) . so no of channels reduced to 256 from 1024.
    

    """ 2.  1x1 conv """ 
    y2 = Conv2D(256, 1, padding="same", use_bias=False)(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)                     #print(y2.shape) = (none, 32,32,256)

    """ 3.  3x3 conv, dilation rate=6 """
    y3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)
    #print(y3.shape)                        #(none, 32,32,256)

    """ 4. 3x3 conv, dilatiom rate=12 """
    y4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)
    #print(y4.shape)                     #(none, 32,32,256)

    """ 5. 3x3 conv, dilation rate=18 """
    y5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(inputs)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)
    #print(y5.shape)                    #(none, 32,32,256)

    """ CONCATENATION AFTER ASPP MODULE """
    y = Concatenate()([y1, y2, y3, y4, y5])  #concatenation
    #print(y.shape)                     #(none,32,32,1280)    #256+256+256+256+256= 1280-no of channels

    """ 1x1 Conv after concatenation in ASPP Module"""
    y = Conv2D(256, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    #print(y.shape)                     #(none, 32, 32, 256)  so, 1x1 convolution reduces the number of channels 

    return y


"""This function defines the main model, which uses ResNet50 as the backbone. 
Features from the fourth block of ResNet50 are fed into the ASPP module. 
Then, the output from the ASPP module and the output from the second block 
of ResNet50 (processed by a 1x1 convolution) are concatenated. 
The result goes through two 3x3 convolutions with Squeeze-and-Excitation blocks after each. 
The final output is a 1x1 convolution with a sigmoid activation function, 
producing pixel-wise classification probabilities."""

def deeplabv3_plus(shape):
    """ Input """
    inputs = Input(shape)

    """ Encoder- Resnet backbone """
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)  #include_top = False as we are not doing classification. We are only doing segmentation.So we dont need last layers of resnet.

    image_features = encoder.get_layer("conv4_block6_out").output   #extract output of resnet's 4th block
    x_a = ASPP(image_features)    #x_a is output from ASPP module              #print(x_a.shape)         #(none, 32, 32, 256) . shape of output from ASPP Module

    """ Bilinear upsampling of the output of ASPP module """
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)                      #print(x_a.shape)           #(none, 128, 128, 256)   

    """Extract low level features from resnet's 2nd layer"""
    x_b = encoder.get_layer("conv2_block2_out").output                             #print(f"x_b shape from encoder: {x_b.shape}")       #(none, 128,128,256)    

    #NOTE: x_a is output of ASPP module. x_b is low-level feature, output of encoder.

    """1x1 CONV after extracting low-level features """
    #There is 1×1 convolution on the low-level features before concatenation to reduce the number of channels,
    # since the corresponding low-level features usually contain a large number of channels (e.g., 256 or 512) 
    #which may outweigh the importance of the rich encoder features.
    
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)    #print(f"x_b shape after 1x1 Conv: {x_b.shape}")   #(None, 128, 128, 48)
    x_b = BatchNormalization()(x_b)                                                 #print(f"x_b shape after batch normalization: {x_b.shape}") #(None, 128, 128, 48)
    x_b = Activation('relu')(x_b)                                                   #print(f"x_b shape after relu activation: {x_b.shape}")  #(None, 128, 128, 48)


    """ Concatenation In DECODER """
    x = Concatenate()([x_a, x_b]) #Concatenate Encoder output(x_a) with low-level features(x_b).  #print(x.shape)   #(None, 128, 128, 304)
    x = SqueezeAndExcite(x)    #channel-wise attention mechanism to improve the performance

    #After the concatenation, we apply a few 3×3 convolutions to refine the features followed by 
    #another simple bilinear upsampling by a factor of 4.

    """ 3x3 Conv After Concatenation in Decoder """
    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)        #3x3 convolution after concatenation in decoder
    x = BatchNormalization()(x)
    x = Activation('relu')(x)                                                        #print(x.shape)    (None, 128, 128, 256)


    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SqueezeAndExcite(x)

    """ LAST BILINEAR UPSAMPLING IN DECODER """
    x = UpSampling2D((4, 4), interpolation="bilinear")(x) #Upsample by 4 after concatenation.    #print(x.shape)     (None, 512, 512, 256)
    x = Conv2D(1, 1)(x)             # 1x1 convolution                                            #print(x.shape) (None, 512, 512, 1)
    x = Activation("sigmoid")(x)    #Last activation function is sigmoid as we are doing binary segmentation.    #print(x.shape)   (None, 512, 512, 1). this is output shape of mask.


    model = Model(inputs, x)
    return model


if __name__ == "__main__":
    input_shape = (512,512,3)
    model =deeplabv3_plus(input_shape)
    model.summary()
