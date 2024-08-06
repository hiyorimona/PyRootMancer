import os
import argparse
import keras
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import ClientSecretCredential
from keras.src.callbacks import EarlyStopping
from patchify import patchify, unpatchify
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.callbacks import EarlyStopping
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Model
import tqdm as tq
from skimage.transform import resize
import logging
import cv2
import numpy as np
from skan import Skeleton, summarize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




@keras.saving.register_keras_serializable(name="f1")
def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

@keras.saving.register_keras_serializable(name="iou")
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total = K.sum(K.square(y_true), [1, 2, 3]) + K.sum(K.square(y_pred), [1, 2, 3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())

    return K.mean(f(y_true, y_pred), axis=-1)

class Modelling():
    def __init__(self):
        self.create_model = self.unet_model((256, 256, 3), 1, 256)
    def unet_model(self, input_shape, num_classes, patch_size):
        inputs = Input(shape=input_shape)
        s = inputs

        # Contraction path
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(patch_size, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        # Adjust the number of output channels to match the number of classes
        outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1, iou])

        return model

    def train_data_generator(self, train_image_folder, train_mask_folder):
        # Training images
        train_image_datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)
        train_image_generator = train_image_datagen.flow_from_directory(
            train_image_folder,
            target_size=(256, 256),
            batch_size=16,
            class_mode=None,  # None since you don't want labels for images
            color_mode='rgb',
            seed=42,
            subset='training'  # specify the subset as 'training'
        )

        # Validation images
        validation_image_generator = train_image_datagen.flow_from_directory(
            train_image_folder,
            target_size=(256, 256),
            batch_size=16,
            class_mode=None,  # None since you don't want labels for images
            color_mode='rgb',
            seed=42,
            subset='validation'  # specify the subset as 'validation'
        )

        # Check if any files are found in the image folder
        if len(train_image_generator.filepaths) == 0 or len(validation_image_generator.filepaths) == 0:
            logging.error(f"No files found in {os.path.basename(train_image_folder)} folder.")
            return None, None, None, None

        # Training masks
        train_mask_datagen = ImageDataGenerator(validation_split=0.2)  # You can also use validation_split here
        train_mask_generator = train_mask_datagen.flow_from_directory(
            train_mask_folder,
            target_size=(256, 256),
            batch_size=16,
            color_mode='grayscale',  # Grayscale for multiclass segmentation
            class_mode=None,
            seed=42,
            subset='training'  # specify the subset as 'training'
        )

        # Validation masks
        validation_mask_generator = train_mask_datagen.flow_from_directory(
            train_mask_folder,
            target_size=(256, 256),
            batch_size=16,
            color_mode='grayscale',  # Grayscale for multiclass segmentation
            class_mode=None,
            seed=42,
            subset='validation'  # specify the subset as 'validation'
        )


        # Check if any files are found in the mask folder
        if len(train_mask_generator.filepaths) == 0 or len(validation_mask_generator.filepaths) == 0:
            logging.error(f"No files found in {os.path.basename(train_mask_folder)} folder and "
                          f"{os.path.basename(train_mask_folder)} ")
            return None, None, None, None


        # Create data generators
        train_generator = self.custom_data_generator(train_image_generator, train_mask_generator)
        validation_generator = self.custom_data_generator(validation_image_generator, validation_mask_generator)

        return train_generator, validation_generator, train_image_generator, validation_image_generator

    def test_data_generator(self, test_image_folder, test_mask_folder):

        test_image_datagen = ImageDataGenerator(rescale=1. / 255)
        test_image_generator = test_image_datagen.flow_from_directory(
            test_image_folder,
            target_size=(256, 256),
            batch_size=16,
            class_mode=None,
            color_mode='rgb',
            seed=42,
        )
        # Training masks
        test_mask_datagen = ImageDataGenerator()
        test_mask_generator = test_mask_datagen.flow_from_directory(
            test_mask_folder,
            target_size=(256, 256),
            batch_size=16,
            color_mode='grayscale',
            class_mode=None,
            seed=42,
        )

        test_generator = self.custom_data_generator(test_image_generator, test_mask_generator)
        return test_generator, test_image_generator


    def custom_data_generator(self, image_generator, mask_generator):
        """
        Custom data generator to yield batches of image and mask pairs.

        Args:
            image_generator (Iterator): Iterator yielding batches of images.
            mask_generator (Iterator): Iterator yielding batches of masks.
        """
        while True:
            try:
                image_batch = next(image_generator)
                mask_batch = next(mask_generator)

                # Check if image_batch and mask_batch have different shapes
                if image_batch.shape[:2] != mask_batch.shape[:2]:
                    logging.error("Image batch and mask batch have different shapes.")
                    continue

                yield image_batch, mask_batch

            except StopIteration as e:
                # If either generator reaches the end, log the error and break the loop
                logging.error("One of the generators reached the end.")
                break

            except Exception as e:
                logging.error(f"Error in custom_data_generator: {e}")
                continue


    def train(self, epochs, train_image_folder, train_mask_folder, model_path, model_name):

        train_generator, validation_generator, train_image_generator, validation_image_generator = (
            self.train_data_generator(train_image_folder, train_mask_folder))

        if None in (train_generator, validation_generator, train_image_generator, validation_image_generator):
            logging.error("One or more generators are None. Training cannot proceed.")
        else:
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')

            history = self.create_model.fit(
                train_generator,
                steps_per_epoch=len(train_image_generator),
                validation_data=validation_generator,
                validation_steps=validation_image_generator.samples,
                epochs=epochs,
                callbacks=early_stopping)

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            self.create_model.save(os.path.join(model_path, f'{model_name}.keras'))

            # mlflow.log_metric("train_f1", history.history["f1"][-1])
            # mlflow.log_metric("train_iou", history.history["iou"][-1])
            # mlflow.log_metric("train_loss", history.history["loss"][-1])
            #
            # mlflow.log_metric("val_f1", history.history["val_f1"][-1])
            # mlflow.log_metric("val_iou", history.history["val_iou"][-1])
            # mlflow.log_metric("val_loss", history.history["val_loss"][-1])


            fig = plt.figure()

            plt.plot(history.history["loss"], label="train loss")
            plt.plot(history.history["val_loss"], label="val loss")
            plt.plot(history.history["f1"], label="train f1")
            plt.plot(history.history["val_f1"], label="val f1")

            plt.title("Training Loss and F1")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/F1")
            plt.legend(loc="lower left")
            # mlflow.log_figure(fig, "metrics.png")


    def load(self, model_path, model_name):
        logging.info(f"loading model from {model_path}")

        model = load_model(os.path.join(model_path, f"{model_name}.keras"))
        logging.info('successful loaded')

        return model

    def register(self, model_path):
        from azure.ai.ml.entities import Model

        subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
        resource_group = "buas-y2"
        workspace_name = "CV5"
        tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
        client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
        client_secret = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        registered_model = Model(
            path=model_path,
            type=AssetTypes.CUSTOM_MODEL,
            name="regmodel",
            description="Model created from pipeline")
        ml_client.models.create_or_update(registered_model)
        print("Model registered.")



    def eval(self, test_image_folder, test_mask_folder, model_path, model_name):
       model = self.load(model_path, model_name)

       test_generator, test_image_generator = self.test_data_generator(test_image_folder, test_mask_folder)
       evaluation = model.evaluate(test_generator, steps=16)
       eval_f1 = evaluation[2]
       eval_iou = evaluation[3]


       # mlflow.log_metric("eval f1", eval_f1)
       # mlflow.log_metric("eval iou", eval_iou)

       threshold = 0.7
       if eval_f1 > threshold or eval_iou > threshold:
           logger.info("Model meets metrics criteria for registering")
           self.register(model_path)
       else:
           logging.info("Model does not meet metrics criteria for registering f1\iou is below 0.8")
    def padder(self, image: np.ndarray) -> np.ndarray:

        patch_size = 256  # Get the patch size from the configuration

        # Get the current dimensions of the image
        h, w = image.shape[:2]
        # Calculate the padding needed for both dimensions
        height_padding = ((h // patch_size) + 1) * patch_size - h
        width_padding = ((w // patch_size) + 1) * patch_size - w

        # Divide the padding evenly between the top and bottom, and left and right
        top_padding = int(height_padding / 2)
        bottom_padding = height_padding - top_padding

        left_padding = int(width_padding / 2)
        right_padding = width_padding - left_padding

        # Add the padding to the image
        padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded_image

    def predict_image(self, image_path: str, output_folder: str, models_folder: str, model_name: str) -> np.ndarray:

        patch_size = 256
        image = cv2.imread(image_path)
        if image.shape[:2] != (2731, 2752):
            normalized_image = cv2.normalize(
                image[75 : image.shape[0] - 200, 750 : image.shape[1] - 700],None,0,255,cv2.NORM_MINMAX,)
            padded_img = self.padder(normalized_image)
        else:
            padded_img = self.padder(image)

        # Patchify the image
        patches = patchify(padded_img, (patch_size, patch_size, 3), step=patch_size)
        patch_x = patches.shape[0]
        patch_y = patches.shape[1]

        # Reshape patches for model prediction
        patches = patches.reshape(-1, patch_size, patch_size, 3)

        # Normalize and predict using the model
        model = self.load(models_folder, model_name)
        logging.info(f"Predicting {os.path.basename(image_path)}")
        preds = model.predict(patches / 255)

        # Reshape predicted patches
        preds = preds.reshape(patch_x, patch_y, patch_size, patch_size)

        # Unpatchify to get the final predicted mask
        predicted_mask = unpatchify(preds, (padded_img.shape[0], padded_img.shape[1]))
        cv2.imwrite(os.path.join(output_folder, f"{os.path.basename(image_path)[:-4]}_predicted_root.png"), predicted_mask)

        return predicted_mask

    def predict(self, input_folder: str, output_folder: str, models_folder: str, model_name: str) -> None:

        test_images_paths = [
            os.path.join(input_folder, file)
            for file in os.listdir(input_folder)]
        for image_path in test_images_paths:
            _ = self.predict_image(image_path, output_folder, models_folder, model_name)

    def opening_closing(self, img: np.array) -> np.array:

        # Create a kernel for morphological operations
        kernel = np.ones((6, 6), dtype="uint8")
        # Apply erosion to the image
        im_erosion = cv2.erode(img, kernel, iterations=1)
        # Apply dilation to the eroded image
        im_dilation = cv2.dilate(im_erosion, kernel, iterations=2)
        # Apply erosion to the dilated image (closing operation)
        im_closing = cv2.erode(im_dilation, kernel, iterations=1)

        return im_closing
    def return_original_size_image(self, image_path: str, output_folder: str) -> np.array:

        original_image = np.zeros((3006, 4202), dtype=np.uint8)
        predicted_mask = cv2.imread(image_path, 0)
        im = predicted_mask[0:2816, 0:2816]
        roi_1 = original_image[50 : 2816 + 50, 720 : 2816 + 720]
        overlay_image = (resize(im, roi_1.shape, mode='reflect', anti_aliasing=True) * 255).astype(np.uint8)
        modified_cropped = np.zeros_like(original_image)
        roi = modified_cropped[50 : 2816 + 50, 720 : 2816 + 720]

        result = cv2.addWeighted(roi, 1, overlay_image, 0.7, 0)
        modified_cropped[50 : 2816 + 50, 720 : 2816 + 720] = result
        norm = cv2.normalize(modified_cropped, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(output_folder, os.path.basename(image_path)), norm)

        return norm

    def revert_size(self, test_folder: str, output_folder: str) -> None:
        test_folder_path = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]
        loop = tq.tqdm(enumerate(test_folder_path),total=len(test_folder_path), bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[38;2;70;130;180m', '\033[0m'),)
        for _, test_path in loop:
            _ = self.return_original_size_image(test_path, output_folder)

        logging.info("Done!")

    # def overlay(self, test_folder, output_folder, model_folder, model_name):
    #     test_paths = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]
    #     predicted_paths = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]
    #     for test_path in test_paths:
    #         _ = self.test_overlaying(test_path, output_folder, model_folder, model_name)


    def overlay(self, test_folder, predicted_folder, output_folder):

        test_folder_paths = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]
        predicted_paths = [os.path.join(predicted_folder, file) for file in os.listdir(predicted_folder)]

        for predicted_path, test_path in zip(predicted_paths, test_folder_paths):
            predicted_mask = cv2.imread(predicted_path,0)
            test_image = cv2.imread(test_path, 0)

            base_img_colored = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
            overlay_img_red = np.zeros_like(base_img_colored)
            overlay_img_red[:, :, 2] = predicted_mask

            # Blend the base image and the overlay image
            blended_img = cv2.addWeighted(base_img_colored, 0.45, overlay_img_red, 1 - 0.45, 0)
            # Resize the blended image for display
            img_resized = cv2.resize(blended_img, (blended_img.shape[1] // 5, blended_img.shape[0] // 5))
            cv2.imwrite(os.path.join(output_folder, os.path.basename(predicted_path)), img_resized)

            return blended_img

    def remove_small_components(self, mask: np.ndarray):

        try:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            sizes = stats[1:, -1]
            im_result = np.zeros_like(labels)
            min_size = 0
            if num_labels < 20:
                min_size = 100
            elif num_labels < 25:
                min_size = 250
            elif num_labels < 35:
                min_size = 300
            elif num_labels < 45:
                min_size = 600
            elif num_labels < 50:
                min_size = 350
            elif num_labels < 55:
                min_size = 500
            elif num_labels < 73:
                min_size = 300
            elif num_labels < 75:
                min_size = 1450
            elif num_labels < 80:
                min_size = 2200
            elif num_labels < 100:
                min_size = 2450
            elif num_labels < 115:
                min_size = 1000
            elif num_labels < 155:
                min_size = 1600
            elif num_labels < 185:
                min_size = 2000
            elif num_labels <= 190:
                min_size = 2300
            for label in range(1, num_labels):
                if sizes[label - 1] >= min_size:
                    im_result[labels == label] = 255

            return im_result.astype(np.uint8)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return np.zeros_like(mask)


    def get_bottom_coordinates(self, image_path, threshold: int = 50):

        try:
            mask = cv2.imread(image_path, 0)
            mask = self.opening_closing(self.remove_small_components(mask))

            # Summarize the skeleton
            df = summarize(Skeleton(mask))
            filtered_df = df[(df['image-coord-dst-0'] < 2415) & (df['branch-type'] != 0) & (df['euclidean-distance'] > 20)]

            # Extract and sort coordinates
            image_coord_dst_0, image_coord_dst_1 = (
                filtered_df.sort_values(by='image-coord-dst-0', ascending=False)
                .groupby('skeleton-id', as_index=False)
                .agg({'image-coord-dst-0': 'max', 'image-coord-dst-1': 'first'})
                [['image-coord-dst-0', 'image-coord-dst-1']]
                .values.T
                .tolist())

            # Pair and sort coordinates
            paired_coords = list(zip(image_coord_dst_0, image_coord_dst_1))
            paired_coords.sort(key=lambda pair: pair[0], reverse=True)
            sorted_image_coord_dst_0, sorted_image_coord_dst_1 = zip(*paired_coords)

            # Convert to lists for further processing
            sorted_image_coord_dst_0 = list(sorted_image_coord_dst_0)
            sorted_image_coord_dst_1 = list(sorted_image_coord_dst_1)

            # Filter the lists based on the threshold
            indices_to_remove = set()
            for i in range(len(sorted_image_coord_dst_1)):
                for j in range(i + 1, len(sorted_image_coord_dst_1)):
                    if (abs(sorted_image_coord_dst_1[i]- sorted_image_coord_dst_1[j]) <= threshold):
                        if (sorted_image_coord_dst_0[i] < sorted_image_coord_dst_0[j]):
                            indices_to_remove.add(i)
                        else:
                            indices_to_remove.add(j)

            # Create new lists excluding the indices to remove
            filtered_image_coord_dst_0 = [
                val for idx, val in enumerate(sorted_image_coord_dst_0) if idx not in indices_to_remove]
            filtered_image_coord_dst_1 = [
                val for idx, val in enumerate(sorted_image_coord_dst_1) if idx not in indices_to_remove]

            # Pair and sort filtered coordinates
            paired_coords = list(zip(filtered_image_coord_dst_0[:5], filtered_image_coord_dst_1[:5]))
            paired_coords.sort(key=lambda pair: pair[1])
            sorted_image_coord_dst_0, sorted_image_coord_dst_1 = zip(
                *paired_coords)

            return sorted_image_coord_dst_0, sorted_image_coord_dst_1

        except Exception as e:
            logging.error(f"No roots detected: {e}")
            return [], []


    def detect(self,test_folder: str, input_folder: str, output_folder: str) -> None:

        # Get paths to the predicted clean masks and original images
        predicted_clean_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
        original_image_paths = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]

        for predicted_clean_path, original_image_path in zip(predicted_clean_paths, original_image_paths):
            image = cv2.imread(original_image_path)
            mask = cv2.imread(predicted_clean_path, 0)

            # Get connected components from the mask
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

            if num_labels == 1:
                img_resized = cv2.resize(image, (image.shape[1] // 5, image.shape[0] // 5))
                # cv2.imshow('Image', 255 - img_resized)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                image_coord_dst_0, image_coord_dst_1 = self.get_bottom_coordinates(predicted_clean_path)
                for i in range(len(image_coord_dst_0)):
                    cv2.circle(image, (image_coord_dst_1[i], image_coord_dst_0[i]), 25, (255, 255, 0), 2)  # Yellow circles

                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_resized = cv2.resize(image_bgr, (image_bgr.shape[1] // 5, image_bgr.shape[0] // 5))
                cv2.imwrite(os.path.join(output_folder,os.path.basename(predicted_clean_path)), img_resized)

                # cv2.imshow('Image', 255 - img_resized)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()



if __name__ == "__main__":
    modelling = Modelling()
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("function", type=str, help="Function to call: 'data_generator' or 'training'")
    parser.add_argument("--train-images-path", type=str, default="data", help="Path to the images")
    parser.add_argument("--train-masks-path", type=str, default="data", help="Path to the masks")
    parser.add_argument("--test-images-path", type=str, default="data", help="Path to the images")
    parser.add_argument("--test-masks-path", type=str, default="data", help="Path to the masks")
    parser.add_argument("--predictions", type=str, default="predicted_roots", help="predicted masks")
    parser.add_argument("--modified-preds", type=str, default="predicted_roots", help="predicted masks")


    parser.add_argument("--epochs", type=int, default=1, help="number of iterations")
    parser.add_argument("--model-path", type=str, default="outputs", help="Path to the trained model")
    parser.add_argument("--model-name", type=str, default="unet_model", help="Model name")


    args = parser.parse_args()


    if args.function == "train":
        modelling.train(args.epochs, args.train_images_path, args.train_masks_path, args.model_path, args.model_name)
    elif args.function == "eval":
        modelling.eval(args.test_images_path, args.test_masks_path, args.model_path, args.model_name)
    elif args.function == "train-data-generator":
        modelling.train_data_generator(args.train_images_path, args.train_masks_path)
    elif args.function == "predict":
        modelling.predict(args.test_images_path, args.predictions, args.model_path, args.model_name)
    elif args.function == 'revert-size':
        modelling.revert_size(args.predictions, args.modified_preds)
    elif args.function == 'overlay':
        modelling.overlay(args.test_images_path, args.predictions, args.modified_preds)
    elif args.function == 'detect':
        modelling.detect(args.test_images_path, args.predictions, args.modified_preds)
    else:
        print(f"Unknown function: {args.function}")


