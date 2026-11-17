import os
# safer runtime settings for Pascal
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from keras.models import Model
from keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate,
    Conv3DTranspose, BatchNormalization, Dropout, Activation,
    Multiply, Add, AveragePooling3D
)
from keras import backend as K
import tensorflow as tf

# force NDHWC globally
K.set_image_data_format("channels_last")

# 3D Dense Inception Residual Block
def dense_inception_res_block(x, filters, dropout):
    conv1 = Conv3D(filters // 4, (1, 1, 1), padding="same", activation="relu", data_format="channels_last")(x)
    conv3 = Conv3D(filters // 4, (3, 3, 3), padding="same", activation="relu", data_format="channels_last")(x)
    conv5 = Conv3D(filters // 4, (5, 5, 5), padding="same", activation="relu", data_format="channels_last")(x)
    concat = concatenate([conv1, conv3, conv5, x])  # dense connection on channels_last
    out = Conv3D(filters, (1, 1, 1), padding="same", data_format="channels_last")(concat)
    x_resized = Conv3D(filters, (1, 1, 1), padding="same", data_format="channels_last")(x)
    out = Add()([out, x_resized])  # residual connection
    out = Activation("relu")(out)
    if dropout > 0:
        out = Dropout(dropout)(out)
    return out

# 3D Attention Gate
def attention_gate(x, g, inter_channels):
    theta_x = Conv3D(inter_channels, (1, 1, 1), padding="same", data_format="channels_last")(x)
    phi_g   = Conv3D(inter_channels, (1, 1, 1), padding="same", data_format="channels_last")(g)
    add_xg  = Add()([theta_x, phi_g])
    act_xg  = Activation("relu")(add_xg)
    psi     = Conv3D(1, (1, 1, 1), activation="sigmoid", padding="same", data_format="channels_last")(act_xg)
    return Multiply()([x, psi])

# Multi-scale Downsampling, now using Keras pooling
def downsample_input_layer(x, scale):
    # VALID matches your Lambda, and 128 is divisible by 2,4,8
    return AveragePooling3D(pool_size=scale, strides=scale, padding="valid", data_format="channels_last")(x)

# 3D Attention Residual U-Net with Multi-Scale Input
def unet_3d(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))

    # multi-scale inputs
    inputs2 = downsample_input_layer(inputs, 2)
    inputs4 = downsample_input_layer(inputs, 4)
    inputs8 = downsample_input_layer(inputs, 8)

    # Contracting path with multi-scale
    c1 = dense_inception_res_block(inputs, 16, dropout=0.1)
    p1 = MaxPooling3D((2, 2, 2), data_format="channels_last")(c1)

    c2 = dense_inception_res_block(concatenate([p1, inputs2]), 32, dropout=0.1)
    p2 = MaxPooling3D((2, 2, 2), data_format="channels_last")(c2)

    c3 = dense_inception_res_block(concatenate([p2, inputs4]), 64, dropout=0.2)
    p3 = MaxPooling3D((2, 2, 2), data_format="channels_last")(c3)

    c4 = dense_inception_res_block(concatenate([p3, inputs8]), 128, dropout=0.2)
    p4 = MaxPooling3D((2, 2, 2), data_format="channels_last")(c4)

    #bottleneck
    c5 = dense_inception_res_block(p4, 256, dropout=0.3)

    # Expanding path with attention
    u6   = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same", data_format="channels_last")(c5)
    att6 = attention_gate(c4, u6, 128)
    u6   = concatenate([u6, att6])
    c6   = dense_inception_res_block(u6, 128, dropout=0.2)

    u7   = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding="same", data_format="channels_last")(c6)
    att7 = attention_gate(c3, u7, 64)
    u7   = concatenate([u7, att7])
    c7   = dense_inception_res_block(u7, 64, dropout=0.2)

    u8   = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding="same", data_format="channels_last")(c7)
    att8 = attention_gate(c2, u8, 32)
    u8   = concatenate([u8, att8])
    c8   = dense_inception_res_block(u8, 32, dropout=0.1)

    u9   = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding="same", data_format="channels_last")(c8)
    att9 = attention_gate(c1, u9, 16)
    u9   = concatenate([u9, att9])
    c9   = dense_inception_res_block(u9, 16, dropout=0.1)

    outputs = Conv3D(num_classes, (1, 1, 1), activation="softmax", padding="same", data_format="channels_last")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Instantiate
if __name__ == "__main__":
    model = unet_3d(128, 128, 128, 3, 4)
    model.summary()
    print("Input:", model.input_shape)
    print("Output:", model.output_shape)
