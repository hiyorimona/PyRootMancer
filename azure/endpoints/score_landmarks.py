import os

import cv2
import keras
import numpy as np
from keras.models import load_model
from patchify import patchify, unpatchify
from skan import Skeleton, summarize
import tensorflow.keras.backend as K

import json
import base64
import io
from PIL import Image
from skimage.transform import resize




# ========== METRICS FOR LOADING THE MODELS
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


#============== LOADING MODEL FROM AZURE
def init():
    # Define the model as a global variable to be used later in the predict function
    global model

    # Get the path where the model is saved, it is set in the environment variable AZUREML_MODEL_DIR by the deployment configuration
    base_path = os.getenv("AZUREML_MODEL_DIR")
    print(f"base_path: {base_path}")

    # show the files in the model_path directory
    print(f"list files in the model_path directory")
    # list files and dirs in the model_path directory
    list_files(base_path)

    # add the model file name to the base_path
    model_path = os.path.join(base_path, 'best_model_root_masks.keras')
    # print the model_path to check if it is correct
    print(f"model_path: {model_path}")

    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully")

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            print("{}{}".format(subindent, f))

#=============== MAIN FUNCTION
def run(raw_data):
    print(f"raw_data: {raw_data}")
    data = json.loads(raw_data)
    print(f"data: {data}")
    base64_image = data["data"]
    print(f"base64_image: {base64_image}")
    image_bytes = base64.b64decode(base64_image)
    print(f"image_bytes: {image_bytes}")
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Convert to RGB
    print("Image loaded successfully with PIL")
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # Grayscale image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        print("Converted grayscale image to RGB")

    predicted_mask = predict_image(image_np, model)
    reverted_mask = revert(predicted_mask)
    landmarks = detect(reverted_mask, image_np)

    _, buffer = cv2.imencode('.png', landmarks)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    return {"result": mask_base64}


#============= PREDICTING FROM INPUT IMAGE

def padder(image: np.ndarray):
    patch_size = 256  # Get the patch size from the configuration
    h, w = image.shape[:2]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w
    top_padding = int(height_padding / 2)
    bottom_padding = height_padding - top_padding
    left_padding = int(width_padding / 2)
    right_padding = width_padding - left_padding
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def predict_image(image, model):
    patch_size = 256
    if image.shape[:2] != (2731, 2752):
        normalized_image = cv2.normalize(
            image[75:image.shape[0] - 200, 750:image.shape[1] - 700],None,0,255,cv2.NORM_MINMAX,)
        padded_img = padder(normalized_image)
    else:
        padded_img = padder(image)
    # Patchify the image
    patches = patchify(padded_img, (patch_size, patch_size, 3), step=patch_size)
    patch_x = patches.shape[0]
    patch_y = patches.shape[1]
    # Reshape patches for model prediction
    patches = patches.reshape(-1, patch_size, patch_size, 3)
    preds = model.predict(patches / 255)
    preds = preds.reshape(patch_x, patch_y, patch_size, patch_size)
    predicted_mask = unpatchify(preds, (padded_img.shape[0], padded_img.shape[1]))
    return predicted_mask



#============ LANDMARKS
def revert(predicted_mask) -> np.array:

    original_image = np.zeros((3006, 4202), dtype=np.uint8)
    im = predicted_mask[0:2816, 0:2816]
    roi_1 = original_image[50 : 2816 + 50, 720 : 2816 + 720]
    overlay_image = (resize(im, roi_1.shape, mode='reflect', anti_aliasing=True) * 255).astype(np.uint8)
    modified_cropped = np.zeros_like(original_image)
    roi = modified_cropped[50 : 2816 + 50, 720 : 2816 + 720]
    result = cv2.addWeighted(roi, 1, overlay_image, 0.7, 0)
    modified_cropped[50 : 2816 + 50, 720 : 2816 + 720] = result
    norm = cv2.normalize(modified_cropped, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm, 128.5, 255, cv2.THRESH_BINARY)

    return thresh


def opening_closing(img):
    kernel = np.ones((6, 6), dtype="uint8")
    im_dilation = cv2.dilate(img, kernel, iterations=2)
    im_closing = cv2.erode(im_dilation, kernel, iterations=1)
    return im_closing


def remove_small_components(mask: np.ndarray) -> np.ndarray:
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
        print(f"An error occurred: {e}")
        return np.zeros_like(mask)

def get_bottom_coordinates(mask, threshold: int = 50):
    try:
        mask = opening_closing(remove_small_components(mask))
        df = summarize(Skeleton(mask))
        filtered_df = df[
            (df['image-coord-dst-0'] < 2415) & (df['branch-type'] != 0) & (df['euclidean-distance'] > 20)
            ]

        image_coord_dst_0, image_coord_dst_1 = (
            filtered_df.sort_values(by='image-coord-dst-0', ascending=False)
            .groupby('skeleton-id', as_index=False)
            .agg({'image-coord-dst-0': 'max', 'image-coord-dst-1': 'first'})[
                ['image-coord-dst-0', 'image-coord-dst-1']
            ]
            .values.T.tolist()
        )

        paired_coords = list(zip(image_coord_dst_0, image_coord_dst_1))
        paired_coords.sort(key=lambda pair: pair[0], reverse=True)
        sorted_image_coord_dst_0, sorted_image_coord_dst_1 = zip(*paired_coords)
        sorted_image_coord_dst_0 = list(sorted_image_coord_dst_0)
        sorted_image_coord_dst_1 = list(sorted_image_coord_dst_1)
        indices_to_remove = set()
        for i in range(len(sorted_image_coord_dst_1)):
            for j in range(i + 1, len(sorted_image_coord_dst_1)):
                if abs(sorted_image_coord_dst_1[i] - sorted_image_coord_dst_1[j]) <= threshold:
                    if sorted_image_coord_dst_0[i] < sorted_image_coord_dst_0[j]:
                        indices_to_remove.add(i)
                    else:
                        indices_to_remove.add(j)
        filtered_image_coord_dst_0 = [
            val for idx, val in enumerate(sorted_image_coord_dst_0) if idx not in indices_to_remove]
        filtered_image_coord_dst_1 = [
            val for idx, val in enumerate(sorted_image_coord_dst_1) if idx not in indices_to_remove]
        paired_coords = list(zip(filtered_image_coord_dst_0[:5], filtered_image_coord_dst_1[:5]))
        paired_coords.sort(key=lambda pair: pair[1])
        sorted_image_coord_dst_0, sorted_image_coord_dst_1 = zip(*paired_coords)

        return sorted_image_coord_dst_0, sorted_image_coord_dst_1

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []

def detect(mask, image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels == 1:
        return image
    else:
        # mask = self.segmentation.opening_closing(self.remove_small_components(mask))
        image_coord_dst_0, image_coord_dst_1 = get_bottom_coordinates(mask)
        print(image_coord_dst_0, image_coord_dst_1)

        # Draw colored circles on the image copy
        for i in range(len(image_coord_dst_0)):
            cv2.circle(
                image, (image_coord_dst_1[i], image_coord_dst_0[i]), 25, (255, 0, 0), 2
            )  # Green color circles
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image_bgr

