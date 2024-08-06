from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import os
import cv2
import keras
import numpy as np
from keras.models import load_model
from patchify import patchify, unpatchify
from skan import Skeleton, summarize
import keras.backend as K
import io
from PIL import Image
from skimage.transform import resize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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

def init():
    global model
    model_path = 'models/best_model_root_masks.keras'
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully")

init()

@app.post("/predict_roots/")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()
        logger.info("File read into memory")
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            logger.info("Converted grayscale image to RGB")

        predicted_mask = predict_image(image_np, model)
        reverted_mask = revert(predicted_mask)
        label = overlay(image_np, reverted_mask)

        _, buffer = cv2.imencode('.png', label)
        logger.info("Prediction completed successfully")
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def padder(image: np.ndarray):
    patch_size = 256
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
            image[75:image.shape[0] - 200, 750:image.shape[1] - 700], None, 0, 255, cv2.NORM_MINMAX,)
        padded_img = padder(normalized_image)
    else:
        padded_img = padder(image)
    patches = patchify(padded_img, (patch_size, patch_size, 3), step=patch_size)
    patch_x = patches.shape[0]
    patch_y = patches.shape[1]
    patches = patches.reshape(-1, patch_size, patch_size, 3)
    preds = model.predict(patches / 255)
    preds = preds.reshape(patch_x, patch_y, patch_size, patch_size)
    predicted_mask = unpatchify(preds, (padded_img.shape[0], padded_img.shape[1]))
    return predicted_mask

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
