from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    concatenate,
)
from keras.models import Model
from src.models.model_evaluation import f1, iou
from typing import Tuple


def unet_model(input_shape: Tuple[int, int, int], num_classes: int, patch_size: int) -> Model:
    """
    Defines and compiles a U-Net model for image segmentation.

    This function creates a U-Net model architecture, which is widely used for image segmentation tasks.
    The model consists of a contracting path to capture context and a symmetric expanding path
    that enables precise localization.

    Parameters:
    -----------
    input_shape : Tuple[int, int, int]
        Shape of the input images (height, width, channels).
    num_classes : int
        Number of classes for the output layer.
    patch_size : int
        Size of the patches to be used in the final convolution layer.

    Returns:
    --------
    Model
        The compiled U-Net model.
    """
    inputs = Input(shape=input_shape)  # Input layer
    s = inputs

    # Contraction path
    c1 = Conv2D(
        16,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(
        16,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        32,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        32,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        128,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        256,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(
        patch_size,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(
        32,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(
        16,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
    )(c9)

    # Adjust the number of output channels to match the number of classes
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)  # Define the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1, iou])  # Compile the model

    return model


def create_model(input_shape: Tuple[int, int, int], num_classes: int, patch_size: int) -> Model:
    """
    Creates and returns a U-Net model for image segmentation.

    Parameters:
    -----------
    input_shape : Tuple[int, int, int]
        Shape of the input images (height, width, channels).
    num_classes : int
        Number of classes for the output layer.
    patch_size : int
        Size of the patches to be used in the final convolution layer.

    Returns:
    --------
    Model
        The compiled U-Net model.
    """
    return unet_model(input_shape, num_classes, patch_size)
