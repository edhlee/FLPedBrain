import numpy as np
import tensorflow as tf
from i3d_inception import Inception_Inflated3d
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization,
    PReLU, Conv3DTranspose, GlobalAveragePooling3D, Dropout, Dense, Concatenate
)
from tensorflow.keras.models import Model

def upsample3d(filters, kernel_size, strides, apply_dropout=False):
    """
    Creates an upsampling layer for 3D data.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    layers = [
        Conv3DTranspose(
            filters=filters, kernel_size=kernel_size, strides=strides,
            padding='same', kernel_initializer=initializer, use_bias=False
        ),
        BatchNormalization(),
        Activation('relu')
    ]

    if apply_dropout:
        layers.append(Dropout(0.5))

    return tf.keras.Sequential(layers)

def unet_model():
    NUM_FRAMES = 64
    FRAME_HEIGHT = 256
    FRAME_WIDTH = 256
    NUM_RGB_CHANNELS = 3

    base_model = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
        classes=5
    )

    layer_names = [
        'Conv3d_1a_7x7', 'MaxPool2d_2a_3x3', 'MaxPool2d_3a_3x3', 
        'MaxPool2d_4a_3x3', 'MaxPool2d_5a_2x2'
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    down_stack = Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = True

    up_stack = [
        upsample3d(832, (2, 3, 3), (2, 2, 2)),
        upsample3d(480, (2, 3, 3), (2, 2, 2)),
        upsample3d(192, (1, 3, 3), (1, 2, 2)),
        upsample3d(64, (1, 3, 3), (1, 2, 2)),
        upsample3d(64, (2, 3, 3), (2, 2, 2))
    ]

    inputs = Input(shape=[NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1])
    x = Concatenate(axis=4)([inputs] * 3)

    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    last = Conv3DTranspose(1, 5, strides=2, padding='same')
    x = last(x)

    feature_map = GlobalAveragePooling3D()(x)
    hidden = Dropout(0.5)(feature_map)
    hidden = Activation('relu')(hidden)
    hidden = Dense(128, name="fc_class")(hidden)
    hidden = Activation('relu')(hidden)
    classifier_output = Dense(5, name="independent_class")(hidden)
    classifier_output = Activation('softmax', dtype='float32', name="class2")(classifier_output)

    output = Activation('sigmoid', dtype='float32')(x)

    return Model(inputs=[inputs], outputs=[output, classifier_output])

# model = unet_model()
