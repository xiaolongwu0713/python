"""
 ARL_EEGModels - A collection of Convolutional Neural Network models for EEG
 Signal Processing and Classification, using Keras and Tensorflow

 Requirements:
    (1) tensorflow-gpu == 1.12.0
    (2) 'image_data_format' = 'channels_first' in keras.json config
    (3) Data shape = (trials, kernels, channels, samples), which for the 
        input layer, will be (trials, 1, channels, samples).
 
 To run the EEG/MEG ERP classification sample script, you will also need

    (4) mne >= 0.17.1
    (5) PyRiemann >= 0.2.5
    (6) scikit-learn >= 0.20.1
    (7) matplotlib >= 2.2.3

"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend
from tensorflow.keras.layers import add

def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (1, Chans, Samples))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (1, Chans, Samples),
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def DeepConvNet6(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3                 1, 3
    strides          1, 2        1, 3                 1, 3
    conv filters     1, 5        1, 10                1, 50
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main   = Input((1, Chans, Samples))
    block1       = Conv2D(64, (1, 10),
                                 input_shape=(1, Chans, Samples),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(64, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(128, (1, 10),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(256, (1, 10),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(512, (1, 10),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet9(nb_classes, Chans=64, Samples=256,
                dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropoutRate)(block1)
##############################################################################
    block2 = Conv2D(128, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Conv2D(128, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2 = Dropout(dropoutRate)(block2)
##############################################################################
    block3 = Conv2D(256, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Conv2D(256, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3 = Dropout(dropoutRate)(block3)
##############################################################################
    block4 = Conv2D(512, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Conv2D(512, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4 = Dropout(dropoutRate)(block4)
##############################################################################
    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet18(nb_classes, Chans=64, Samples=256,
                dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropoutRate)(block1)
##############################################################################
    block2 = Conv2D(128, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Conv2D(128, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Conv2D(128, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Conv2D(128, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Conv2D(128, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2 = Dropout(dropoutRate)(block2)
##############################################################################
    block3 = Conv2D(256, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Conv2D(256, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Conv2D(256, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Conv2D(256, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Conv2D(256, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3 = Dropout(dropoutRate)(block3)
##############################################################################
    block4 = Conv2D(512, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Conv2D(512, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Conv2D(512, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Conv2D(512, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Conv2D(512, (1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4 = Dropout(dropoutRate)(block4)
##############################################################################
    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

# need these for ShallowConvNet
def square(x):
    return backend.square(x)

def log(x):
    return backend.log(backend.clip(x, min_value = 1e-7, max_value = 10000))


def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """

    # start the model
    input_main   = Input((1, Chans, Samples))
    block1       = Conv2D(64, (1, 25),
                                 input_shape=(1, Chans, Samples),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(64, (Chans, 1), use_bias=False,
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 75), strides=(1, 75))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = backend.int_shape(input)
    residual_shape = backend.int_shape(residual)
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    stride_height = int(round(input_shape[3] / residual_shape[3]))
    equal_channels = input_shape[1] == residual_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[1],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                         )(input)

    return add([shortcut, residual])


def ResDeepConvNet9(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = Dropout(dropoutRate)(block2)

    block2 = Conv2D(128, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    # block2 = _shortcut(block1, block2)
##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = Dropout(dropoutRate)(block3)

    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    # block3 = _shortcut(block2, block3)
##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 3),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = Dropout(dropoutRate)(block4)

    block4 = Conv2D(512, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    # block4 = _shortcut(block3, block4)
##############################################################################

    # flatten = Flatten()(block5)
    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_128_10(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################

    block2 = Dropout(dropoutRate)(block2)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block2)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)



def DeepConvNet_210519_256_10(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block3 = Dropout(dropoutRate)(block3)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block3)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet_210519_1024_10(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block5 = Dropout(dropoutRate)(block4)
    block5 = Conv2D(1024, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block5)
    block5 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block5)
    block5 = Activation('elu')(block5)

    ##############################################################################

    block5 = Dropout(dropoutRate)(block5)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block5)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)



def DeepConvNet_210519_2048_10(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block5 = Dropout(dropoutRate)(block4)
    block5 = Conv2D(1024, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block5)
    block5 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block5)
    block5 = Activation('elu')(block5)

    ##############################################################################

    block6 = Dropout(dropoutRate)(block5)
    block6 = Conv2D(2048, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block6)
    block6 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block6)
    block6 = Activation('elu')(block6)

    ##############################################################################

    block6 = Dropout(dropoutRate)(block6)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block6)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet_210519_4096_10(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block5 = Dropout(dropoutRate)(block4)
    block5 = Conv2D(1024, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block5)
    block5 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block5)
    block5 = Activation('elu')(block5)

    ##############################################################################

    block6 = Dropout(dropoutRate)(block5)
    block6 = Conv2D(2048, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block6)
    block6 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block6)
    block6 = Activation('elu')(block6)

    ##############################################################################

    block7 = Dropout(dropoutRate)(block6)
    block7 = Conv2D(4096, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block7)
    block7 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block7)
    block7 = Activation('elu')(block7)

    ##############################################################################

    block7 = Dropout(dropoutRate)(block7)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block7)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)



def DeepConvNet_210519_512_6(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 6),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)

    block1 = Conv2D(64, (1, 6),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)

    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 6), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Dropout(dropoutRate)(block2)
    block2 = Conv2D(128, (1, 6), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 6), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 6), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 6), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 6), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_relu(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('relu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('relu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('relu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('relu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_2222(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Dropout(dropoutRate)(block2)
    block2 = Conv2D(128, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_2222_res(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Dropout(dropoutRate)(block2)
    block2 = Conv2D(128, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)

    block2 = _shortcut(block1, block2)
    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block3 = _shortcut(block2, block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)

    block4 = _shortcut(block3, block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet_210519_512_10_nodropout(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet_210519_512_10_2222_res_nodropout(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    # block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    # block2 = Dropout(dropoutRate)(block2)
    block2 = Conv2D(128, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)

    block2 = _shortcut(block1, block2)
    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    # block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    # block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block3 = _shortcut(block2, block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    # block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    # block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)

    block4 = _shortcut(block3, block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    # block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet_210519_512_10_343_res(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Dropout(dropoutRate)(block2)
    block2 = Conv2D(128, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Dropout(dropoutRate)(block2)
    block2 = Conv2D(128, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)

    block2 = _shortcut(block1, block2)
    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block3 = _shortcut(block2, block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)

    block4 = _shortcut(block3, block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_343(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Dropout(dropoutRate)(block2)
    block2 = Conv2D(128, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = Dropout(dropoutRate)(block2)
    block2 = Conv2D(128, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)


    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_flatten(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_res(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)

    block2 = _shortcut(block1, block2)
    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block3 = _shortcut(block2, block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)


    block4 = _shortcut(block3, block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def _SE_block(x, filters):
    gap = x
    gap = GlobalAveragePooling2D(data_format='channels_first')(gap)
    gap = tf.reshape(gap, [-1, filters, 1, 1])
    gap = Conv2D(filters // 2, kernel_size=1)(gap)
    gap = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(gap)
    gap = Activation('elu')(gap)
    gap = Conv2D(filters, kernel_size=1)(gap)
    gap = Activation("sigmoid")(gap)
    out = x * gap
    return out


def DeepConvNet_210519_512_10_res_SE(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)

    block2 = _SE_block(block2, 128)
    block2 = _shortcut(block1, block2)
    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block3 = _SE_block(block3, 256)
    block3 = _shortcut(block2, block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)

    block4 = _SE_block(block4, 512)
    block4 = _shortcut(block3, block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_res_SE_2(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = _SE_block(block2, 128)

    ##############################################################################

    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = _SE_block(block3, 256)


    ##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = _SE_block(block4, 512)


    ##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


class GroupedConv2D(object):
    """Groupped convolution.
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_py
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel size.
    """

    def __init__(self, filters, kernel_size, use_keras=True, **kwargs):
        """Initialize the layer.
        Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel for each group.
        use_keras: An boolean value, whether to use keras layer.
        **kwargs: other parameters passed to the original conv2d layer.
        """
        self._groups = len(kernel_size)
        self._channel_axis = 1

        self._convs = []
        splits = self._split_channels(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(self._get_conv2d(splits[i], kernel_size[i], use_keras, **kwargs))

    def _get_conv2d(self, filters, kernel_size, use_keras, **kwargs):
        """A helper function to create Conv2D layer."""
        if use_keras:
            return Conv2D(filters=filters, kernel_size=kernel_size, padding="same", kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format="channels_first")
        else:
            return Conv2D(filters=filters, kernel_size=kernel_size, padding="same", kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format="channels_first")

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        if tf.__version__ < "2.0.0":
            filters = inputs.shape[self._channel_axis].value
        else:
            filters = inputs.shape[self._channel_axis]
        splits = self._split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x

def DeepConvNet_210519_512_10_res_Grouped(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 1), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = GroupedConv2D(filters=128, kernel_size=[(1, 10) for i in range(2)])(block2)

    block2 = _shortcut(block1, block2)
    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 1), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = GroupedConv2D(filters=256, kernel_size=[(1, 10) for i in range(2)])(block3)

    block3 = _shortcut(block2, block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 1), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = GroupedConv2D(filters=64, kernel_size=[(1, 10) for i in range(2)])(block4)

    block4 = _shortcut(block3, block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet_210519_512_10_SpaFst(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(Chans, (Chans, 1), )(input_main)
    block1 = Permute((2,1,3), input_shape=(Chans, 1, Samples))(block1)
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_SpaFst_64(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (Chans, 1), )(input_main)
    block1 = Permute((2,1,3), input_shape=(64, 1, Samples))(block1)
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, 64, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = Conv2D(64, (64, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_SpaFst_Dpth(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(Chans, (Chans, 1), name='conv', )(input_main)
    block1 = Permute((2,1,3), input_shape=(Chans, 1, Samples))(block1)
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    # block1 = Conv2D(64, (Chans, 1),
    #                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=1, name='depth',
                             depthwise_constraint=max_norm(2., axis=(0, 1, 2),))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes,  name='den', kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_SpaFst_Dpth_64(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (Chans, 1), )(input_main)
    block1 = Permute((2,1,3), input_shape=(64, 1, Samples))(block1)
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, 64, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    # block1 = Conv2D(64, (Chans, 1),
    #                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = DepthwiseConv2D((64, 1), use_bias=False,
                             depth_multiplier=1,
                             depthwise_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_SpaFst_Dpth_32(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(32, (Chans, 1), )(input_main)
    block1 = Permute((2,1,3), input_shape=(32, 1, Samples))(block1)
    block1 = Conv2D(32, (1, 10),
                    input_shape=(1, 32, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    # block1 = Conv2D(64, (Chans, 1),
    #                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = DepthwiseConv2D((32, 1), use_bias=False,
                             depth_multiplier=1,
                             depthwise_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_SpaFst_Dpth_res(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(Chans, (Chans, 1), )(input_main)
    block1 = Permute((2,1,3), input_shape=(Chans, 1, Samples))(block1)
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    # block1 = Conv2D(64, (Chans, 1),
    #                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=1,
                             depthwise_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


    ##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)

    block2 = _shortcut(block1, block2)
    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block3 = _shortcut(block2, block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)


    block4 = _shortcut(block3, block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_Deth(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    # block1 = Conv2D(64, (Chans, 1),
    #                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=1,
                             depthwise_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet_210519_512_10_SpaFst_Dpth_MaxPooling(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(Chans, (Chans, 1), name='conv', )(input_main)
    block1 = Permute((2,1,3), input_shape=(Chans, 1, Samples))(block1)
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    # block1 = Conv2D(64, (Chans, 1),
    #                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=1, name='depth',
                             depthwise_constraint=max_norm(2., axis=(0, 1, 2),))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)


##############################################################################
    block3 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)


    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

##############################################################################

    block4 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

##############################################################################

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalMaxPooling2D(name='max_pool')(block4)
    dense = Dense(nb_classes,  name='den', kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_SpaFst_Dpth_res_preactivation(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(Chans, (Chans, 1), )(input_main)
    block1 = Permute((2,1,3), input_shape=(Chans, 1, Samples))(block1)
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    # block1 = Conv2D(64, (Chans, 1),
    #                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=1,
                             depthwise_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropoutRate)(block1)

    ##############################################################################
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)

    block2 = _shortcut(block1, block2)

    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = Dropout(dropoutRate)(block2)

    ##############################################################################
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)

    block3 = _shortcut(block2, block3)

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = Dropout(dropoutRate)(block3)

    ##############################################################################
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block4 = _shortcut(block3, block4)

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = Dropout(dropoutRate)(block4)

    ##############################################################################

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_210519_512_10_res_Grouped_noRes(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 1), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block2 = GroupedConv2D(filters=128, kernel_size=[(1, 10) for i in range(2)])(block2)

    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 1), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block3 = GroupedConv2D(filters=256, kernel_size=[(1, 10) for i in range(2)])(block3)


    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 1), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)
    block4 = GroupedConv2D(filters=64, kernel_size=[(1, 10) for i in range(2)])(block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)