import os

import cv2
import keras
import numpy as np
from keras.models import load_model
from patchify import patchify, unpatchify
from skan import Skeleton, summarize
import keras.backend as K

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
    label = overlay(image_np,reverted_mask)

    _, buffer = cv2.imencode('.png', label)
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

    return norm


def overlay(test_image, predicted_mask):

    base_img_colored = test_image
    overlay_img_red = np.zeros_like(base_img_colored)
    overlay_img_red[:, :, 2] = predicted_mask

    blended_img = cv2.addWeighted(base_img_colored, 0.45, overlay_img_red, 1 - 0.45, 0)

    return blended_img