# src/unet_model.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
import tensorflow as tf

INPUT_SHAPE = (128,128,1)

def build_unet(input_shape=INPUT_SHAPE):
    inputs = Input(shape=input_shape, name="input_layer")
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)

    u1 = Conv2DTranspose(32, 2, strides=2, padding='same')(c3)
    concat1 = Concatenate()([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(concat1)

    u2 = Conv2DTranspose(16, 2, strides=2, padding='same')(c4)
    concat2 = Concatenate()([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(concat2)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

MODEL_WEIGHTS_PATH = "src/model_weights.h5"

def load_unet_model():
    model = build_unet()
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model
