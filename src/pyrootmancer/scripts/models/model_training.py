import logging
import os
import cv2
import tqdm
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from patchify import patchify, unpatchify
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pyrootmancer.scripts.data.data_preprocessing import DataPipelineSetup
from pyrootmancer.scripts.models.model_definitions import unet_model
from pyrootmancer.scripts.models.model_evaluation import iou, f1
from pyrootmancer.scripts.utils.configuration import *
import tensorflow as tf
from typing import Any, Tuple, Iterator
from tensorflow.keras.callbacks import Callback


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class ModelTraining:
    def __init__(self) -> None:
        """Initialize the ModelTraining class and define the U-Net model.

        Author: Jakub Cyba (223860)
        """
        self.unet_model = unet_model(
            param_config.get("input_shape"), param_config.get("num_classes"), param_config.get("patch_size"), 'adam'
        )

    def data_generator(
        self, image_folder: str, mask_folder: str
    ) -> Tuple[Iterator, Iterator, ImageDataGenerator, ImageDataGenerator]:
        """Generate data for training and validation.

        This function expects a mask folder containing mask images. It generates data
        for training and validation by creating image and mask generators from the provided
        directories.

        Args:
            image_folder (str): Path to the folder containing image data.
            mask_folder (str): Path to the folder containing mask images.

        Returns:
            tuple: A tuple containing generators for training and validation images and masks.

        Author: Jakub Cyba (223860)
        """
        train_image_datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0 / 255)
        train_image_generator = train_image_datagen.flow_from_directory(
            image_folder,
            target_size=(param_config.get("patch_size"), param_config.get("patch_size")),
            batch_size=16,
            class_mode=None,
            color_mode='rgb',
            seed=42,
            subset='training',
        )

        validation_image_generator = train_image_datagen.flow_from_directory(
            image_folder,
            target_size=(param_config.get("patch_size"), param_config.get("patch_size")),
            batch_size=16,
            class_mode=None,
            color_mode='rgb',
            seed=42,
            subset='validation',
        )

        if len(train_image_generator.filepaths) == 0 or len(validation_image_generator.filepaths) == 0:
            logging.error(f"No files found in {os.path.basename(image_folder)} folder.")
            return None, None, None, None

        train_mask_datagen = ImageDataGenerator(validation_split=0.2)
        train_mask_generator = train_mask_datagen.flow_from_directory(
            mask_folder,
            target_size=(param_config.get("patch_size"), param_config.get("patch_size")),
            batch_size=16,
            color_mode='grayscale',
            class_mode=None,
            seed=42,
            subset='training',
        )

        validation_mask_generator = train_mask_datagen.flow_from_directory(
            mask_folder,
            target_size=(param_config.get("patch_size"), param_config.get("patch_size")),
            batch_size=16,
            color_mode='grayscale',
            class_mode=None,
            seed=42,
            subset='validation',
        )

        if len(train_mask_generator.filepaths) == 0 or len(validation_mask_generator.filepaths) == 0:
            logging.error(f"No files found in {os.path.basename(mask_folder)} folder.")
            return None, None, None, None

        train_generator = self.custom_data_generator(train_image_generator, train_mask_generator)
        validation_generator = self.custom_data_generator(validation_image_generator, validation_mask_generator)

        return train_generator, validation_generator, train_image_generator, validation_image_generator

    def custom_data_generator(self, image_generator: Iterator, mask_generator: Iterator) -> Iterator:
        """Custom data generator to yield batches of image and mask pairs.

        Args:
            image_generator (Iterator): Iterator yielding batches of images.
            mask_generator (Iterator): Iterator yielding batches of masks.

        Yields:
            Iterator: An iterator yielding tuples of (image_batch, mask_batch).

        Author: Jakub Cyba (223860)
        """
        while True:
            try:
                image_batch = next(image_generator)
                mask_batch = next(mask_generator)

                if image_batch.shape[:2] != mask_batch.shape[:2]:
                    logging.error("Image batch and mask batch have different shapes.")
                    continue

                yield image_batch, mask_batch

            except StopIteration:
                logging.error("One of the generators reached the end.")
                break

            except Exception as e:
                logging.error(f"Error in custom_data_generator: {e}")
                continue

    def training(
        self,
        epochs: int,
        image_folder: str,
        mask_folder: str,
        model_folder: str,
        model_name: str,
        patience: int = 3,
        optimizer: str = "adam",
    ) -> Any:
        """Train the model using the provided mask folder.

        This method trains the U-Net model using the mask images located in the specified folder.
        It prepares data generators for training and validation, fits the model to the training data,
        and evaluates its performance on the validation set.

        Args:
            epochs (int): Number of epochs to train the model.
            image_folder (str): Path to the folder containing image data.
            mask_folder (str): Path to the folder containing mask images for training.
            model_folder (str): Path to the folder where the trained model will be saved.
            model_name (str): Name of the model to save.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            optimizer (str): Optimizer to use for training.

        Returns:
            Any: The trained model.

        Author: Jakub Cyba (223860)
        """
        train_generator, validation_generator, train_image_generator, validation_image_generator = self.data_generator(
            image_folder, mask_folder
        )

        if None in (train_generator, validation_generator, train_image_generator, validation_image_generator):
            logging.error("One or more generators are None. Training cannot proceed.")
            return None
        else:
            model_checkpoint = ModelCheckpoint(os.path.join(model_folder, f'{model_name}.keras'), save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, mode='min')

            history = self.unet_model.fit(
                train_generator,
                steps_per_epoch=len(train_image_generator),
                validation_data=validation_generator,
                validation_steps=validation_image_generator.samples // 16,
                epochs=epochs,
                callbacks=[early_stopping, model_checkpoint],
            )

            return self.unet_model

    def load_model(self, models_folder: str, model_name: str) -> Any:
        """Load a trained model.

        Args:
            models_folder (str): Path to the folder containing the models.
            model_name (str): Name of the model to load.

        Returns:
            Any: The loaded model.

        Author: Jakub Cyba (223860)
        """
        model = load_model(os.path.join(models_folder, f"{model_name}.keras"), custom_objects={'f1': f1, 'iou': iou})
        return model

    def predict_image(
        self, image_path: str, output_folder: str = None, models_folder: str = None, model_name: str = None
    ) -> np.ndarray:
        """Predict the mask for an input image using a given model.

        Args:
            image_path (str): Path to the input image.
            output_folder (str, optional): Path to the folder to save the predicted mask. Defaults to None.
            models_folder (str, optional): Path to the folder containing the models. Defaults to None.
            model_name (str, optional): Name of the model to use for prediction. Defaults to None.

        Returns:
            np.ndarray: Predicted mask for the input image.

        Author: Jakub Cyba (223860)
        """
        processor = DataPipelineSetup()
        patch_size = param_config.get("patch_size")

        image = cv2.imread(image_path)
        if image.shape[:2] != (2731, 2752):
            normalized_image = cv2.normalize(
                image[75 : image.shape[0] - 200, 750 : image.shape[1] - 700], None, 0, 255, cv2.NORM_MINMAX
            )
            padded_img = processor.padder(normalized_image)
        else:
            padded_img = processor.padder(image)

        patches = patchify(padded_img, (patch_size, patch_size, 3), step=patch_size)
        patch_x = patches.shape[0]
        patch_y = patches.shape[1]

        patches = patches.reshape(-1, patch_size, patch_size, 3)

        model = self.load_model(models_folder, model_name)
        logging.info(f"Predicting {os.path.basename(image_path)}")
        preds = model.predict(patches / 255)

        preds = preds.reshape(patch_x, patch_y, patch_size, patch_size)

        predicted_mask = unpatchify(preds, (padded_img.shape[0], padded_img.shape[1]))
        if output_folder:
            cv2.imwrite(
                os.path.join(output_folder, f"{os.path.basename(image_path)[:-4]}_predicted_root.png"), predicted_mask
            )

        return predicted_mask

    def predict_folder(self, input_folder: str, output_folder: str, models_folder: str, model_name: str) -> None:
        """Predict the masks for the test set using a given model.

        Args:
            input_folder (str): Path to the folder containing the input images.
            output_folder (str): Path to the folder to save the predicted masks.
            models_folder (str): Path to the folder containing the models.
            model_name (str): Name of the model to use for prediction.

        Returns:
            None

        Author: Jakub Cyba (223860)
        """
        test_images_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]

        loop = tqdm.tqdm(
            enumerate(test_images_paths),
            total=len(test_images_paths),
            bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[38;2;70;130;180m', '\033[0m'),
        )

        for _, image_path in loop:
            _ = self.predict_image(image_path, output_folder, models_folder, model_name)


if __name__ == "__main__":
    modelling = ModelTraining()
    modelling.training(
        1,
        os.path.dirname(folder_config.get("images_folder_patched")),
        os.path.dirname(folder_config.get("root_folder_patched")),
        folder_config.get("models_folder"),
        'model_name',
    )
