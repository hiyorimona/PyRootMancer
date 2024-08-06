import logging
import os
from typing import Any, Iterator, Tuple

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from patchify import patchify, unpatchify
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from prc.data.data_preprocessing import DataPipelineSetup
from prc.models.model_definitions import unet_model
from prc.models.model_evaluation import iou, f1
from prc.utils.configuration import *

# Configure logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S'
)


class ModelTraining:
    """
    A class for training and evaluating a U-Net model for image segmentation tasks.

    This class provides methods for setting up data generators, training the U-Net model,
    loading a trained model, and making predictions on images or folders of images. It supports
    custom data generation, handles image patching, and applies various training callbacks for
    improved model performance.

    Author: Jakub Cyba
    """

    def __init__(self):
        """
        Initializes the ModelTraining class and defines the U-Net model.
        """
        # Define U-Net model
        self.unet_model = unet_model(
            param_config.get("input_shape"),
            param_config.get("num_classes"),
            param_config.get("patch_size")
        )

    def data_generator(self, image_folder: str, mask_folder: str) -> Tuple[Iterator, Iterator, ImageDataGenerator, ImageDataGenerator]:
        """
        Generate data for training and validation.

        This function expects a mask folder containing mask images. It generates data
        for training and validation by creating image and mask generators from the provided directories.

        Parameters:
        -----------
        image_folder : str
            Path to the folder containing images.
        mask_folder : str
            Path to the folder containing mask images.

        Returns:
        --------
        Tuple
            A tuple containing generators for training and validation images and masks.
        """
        # Training images
        train_image_datagen = ImageDataGenerator(
            validation_split=0.2, rescale=1.0 / 255
        )
        train_image_generator = train_image_datagen.flow_from_directory(
            image_folder,
            target_size=(
                param_config.get("patch_size"),
                param_config.get("patch_size"),
            ),
            batch_size=16,
            class_mode=None,
            color_mode='rgb',
            seed=42,
            subset='training'
        )

        # Validation images
        validation_image_generator = train_image_datagen.flow_from_directory(
            image_folder,
            target_size=(
                param_config.get("patch_size"),
                param_config.get("patch_size"),
            ),
            batch_size=16,
            class_mode=None,
            color_mode='rgb',
            seed=42,
            subset='validation'
        )

        # Check if any files are found in the image folder
        if (
            len(train_image_generator.filepaths) == 0
            or len(validation_image_generator.filepaths) == 0
        ):
            logging.error(
                f"No files found in {os.path.basename(image_folder)} folder."
            )
            return None, None, None, None

        # Training masks
        train_mask_datagen = ImageDataGenerator(validation_split=0.2)
        train_mask_generator = train_mask_datagen.flow_from_directory(
            mask_folder,
            target_size=(
                param_config.get("patch_size"),
                param_config.get("patch_size"),
            ),
            batch_size=16,
            color_mode='grayscale',
            class_mode=None,
            seed=42,
            subset='training'
        )

        # Validation masks
        validation_mask_generator = train_mask_datagen.flow_from_directory(
            mask_folder,
            target_size=(
                param_config.get("patch_size"),
                param_config.get("patch_size"),
            ),
            batch_size=16,
            color_mode='grayscale',
            class_mode=None,
            seed=42,
            subset='validation'
        )

        # Check if any files are found in the mask folder
        if (
            len(train_mask_generator.filepaths) == 0
            or len(validation_mask_generator.filepaths) == 0
        ):
            logging.error(
                f"No files found in {os.path.basename(mask_folder)} folder."
            )
            return None, None, None, None

        # Create data generators
        train_generator = self.custom_data_generator(
            train_image_generator, train_mask_generator
        )
        validation_generator = self.custom_data_generator(
            validation_image_generator, validation_mask_generator
        )

        return (
            train_generator,
            validation_generator,
            train_image_generator,
            validation_image_generator,
        )

    def custom_data_generator(self, image_generator: Iterator, mask_generator: Iterator) -> Iterator:
        """
        Custom data generator to yield batches of image and mask pairs.

        Parameters:
        -----------
        image_generator : Iterator
            Iterator yielding batches of images.
        mask_generator : Iterator
            Iterator yielding batches of masks.

        Yields:
        -------
        Iterator
            Iterator yielding batches of image and mask pairs.
        """
        while True:
            try:
                image_batch = next(image_generator)
                mask_batch = next(mask_generator)

                # Check if image_batch and mask_batch have different shapes
                if image_batch.shape[:2] != mask_batch.shape[:2]:
                    logging.error(
                        "Image batch and mask batch have different shapes."
                    )
                    continue

                yield image_batch, mask_batch

            except StopIteration:
                logging.error("One of the generators reached the end.")
                break

            except Exception as e:
                logging.error(f"Error in custom_data_generator: {e}")
                continue

    def training(self, epochs: int, image_folder: str, mask_folder: str, model_folder: str, model_name: str) -> Any:
        """
        Train the model using the provided image and mask folders.

        This method trains the U-Net model using the images and mask images located in the specified folders.
        It prepares data generators for training and validation, fits the model to the training data,
        and evaluates its performance on the validation set.

        Parameters:
        -----------
        epochs : int
            Number of epochs to train the model.
        image_folder : str
            Path to the folder containing images for training.
        mask_folder : str
            Path to the folder containing mask images for training.
        model_folder : str
            Path to the folder to save the trained model.
        model_name : str
            Name to save the trained model as.

        Returns:
        --------
        Any
            The trained model.
        """
        try:
            train_generator, validation_generator, train_image_generator, validation_image_generator = self.data_generator(
                image_folder, mask_folder)

            if None in (train_generator, validation_generator, train_image_generator, validation_image_generator):
                logging.error("One or more generators are None. Training cannot proceed.")
                return None

            model_checkpoint = ModelCheckpoint(os.path.join(model_folder, f'{model_name}.keras'), save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')

            try:
                history = self.unet_model.fit(
                    train_generator,
                    steps_per_epoch=len(train_image_generator),
                    validation_data=validation_generator,
                    validation_steps=validation_image_generator.samples // 16,
                    epochs=epochs,
                    callbacks=[early_stopping, model_checkpoint]
                )

                return self.unet_model

            except Exception as e:
                logging.error(f"Error during model training: {e}")
                return None

        except ValueError as ve:
            logging.error(f"ValueError during training: {ve}")
        except RuntimeError as re:
            logging.error(f"RuntimeError during training: {re}")
        except IOError as ie:
            logging.error(f"IOError during training: {ie}")
        except Exception as e:
            logging.error(f"Error during training: {e}")
        finally:
            pass

    def load_model(self, models_folder: str, model_name: str) -> Any:
        """
        Load a trained model.

        Parameters:
        -----------
        models_folder : str
            Path to the folder containing the trained models.
        model_name : str
            Name of the model to load.

        Returns:
        --------
        Any
            The loaded model.
        """
        model = load_model(os.path.join(models_folder, f"{model_name}.keras"), custom_objects={'f1': f1, 'iou': iou})
        return model

    def predict_image(self, image_path: str, output_folder: str, models_folder: str, model_name: str) -> np.ndarray:
        """
        Predict the mask for an input image using a given model.

        This function loads the model, preprocesses the input image, predicts the mask,
        and saves the predicted mask.

        Parameters:
        -----------
        image_path : str
            Path to the input image.
        output_folder : str
            Path to the folder to save the predicted mask.
        models_folder : str
            Path to the folder containing the trained models.
        model_name : str
            Name of the model to use for prediction.

        Returns:
        --------
        np.ndarray
            The predicted mask for the input image.
        """
        # Pad the input image to ensure it's compatible with patching
        processor = DataPipelineSetup()
        patch_size = param_config.get("patch_size")

        image = cv2.imread(image_path)
        if image.shape[:2] != (2731, 2752):
            normalized_image = cv2.normalize(
                image[75: image.shape[0] - 200, 750: image.shape[1] - 700],
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            )
            padded_img = processor.padder(normalized_image)
        else:
            padded_img = processor.padder(image)

        # Patchify the image
        patches = patchify(padded_img, (patch_size, patch_size, 3), step=patch_size)
        patch_x = patches.shape[0]
        patch_y = patches.shape[1]

        # Reshape patches for model prediction
        patches = patches.reshape(-1, patch_size, patch_size, 3)

        # Normalize and predict using the model
        model = self.load_model(models_folder, model_name)
        logging.info(f"Predicting {os.path.basename(image_path)}")
        preds = model.predict(patches / 255)

        # Reshape predicted patches
        preds = preds.reshape(patch_x, patch_y, patch_size, patch_size)

        # Unpatchify to get the final predicted mask
        predicted_mask = unpatchify(preds, (padded_img.shape[0], padded_img.shape[1]))
        cv2.imwrite(os.path.join(output_folder,
                    f"{os.path.basename(image_path)[:-4]}_predicted_root.png"), predicted_mask)

        return predicted_mask

    def predict_folder(self, input_folder: str, output_folder: str, models_folder: str, model_name: str) -> None:
        """
        Predict the masks for all images in a folder using a given model.

        This function iterates over all images in the input folder, predicts masks,
        and saves the predicted masks to the output folder.

        Parameters:
        -----------
        input_folder : str
            Path to the folder containing input images.
        output_folder : str
            Path to the folder to save the predicted masks.
        models_folder : str
            Path to the folder containing the trained models.
        model_name : str
            Name of the model to use for prediction.

        Returns:
        --------
        None
        """
        test_images_paths = [
            os.path.join(input_folder, file)
            for file in os.listdir(input_folder)
        ]
        for image_path in test_images_paths:
            _ = self.predict_image(
                image_path, output_folder, models_folder, model_name
            )
