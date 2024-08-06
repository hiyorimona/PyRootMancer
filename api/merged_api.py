from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Define the origins that should be allowed to make CORS requests
origins = [
    "http://localhost:8000",
]

# Add the CORS middleware to the FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


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
    model_path = '../models/best_model_root_masks.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading model from: {model_path}")
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


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
    padded_image = cv2.copyMakeBorder(
        image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded_image


def predict_image(image, model):
    patch_size = 256
    if image.shape[:2] != (2731, 2752):
        normalized_image = cv2.normalize(
            image[75 : image.shape[0] - 200, 750 : image.shape[1] - 700],
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )
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


# ========== PREDICTING FROM INPUT IMAGE


def predict_image_root(image, model):
    patch_size = 256
    if image.shape[:2] != (2731, 2752):
        normalized_image = cv2.normalize(
            image[75 : image.shape[0] - 200, 750 : image.shape[1] - 700], None, 0, 255, cv2.NORM_MINMAX
        )
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
        filtered_df = df[(df['image-coord-dst-0'] < 2415) & (df['branch-type'] != 0) & (df['euclidean-distance'] > 20)]
        image_coord_dst_0, image_coord_dst_1 = (
            filtered_df.sort_values(by='image-coord-dst-0', ascending=False)
            .groupby('skeleton-id', as_index=False)
            .agg({'image-coord-dst-0': 'max', 'image-coord-dst-1': 'first'})[['image-coord-dst-0', 'image-coord-dst-1']]
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
            val for idx, val in enumerate(sorted_image_coord_dst_0) if idx not in indices_to_remove
        ]
        filtered_image_coord_dst_1 = [
            val for idx, val in enumerate(sorted_image_coord_dst_1) if idx not in indices_to_remove
        ]
        paired_coords = list(zip(filtered_image_coord_dst_0[:5], filtered_image_coord_dst_1[:5]))
        paired_coords.sort(key=lambda pair: pair[1])
        sorted_image_coord_dst_0, sorted_image_coord_dst_1 = zip(*paired_coords)
        return list(sorted_image_coord_dst_0), list(sorted_image_coord_dst_1)
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []


def detect(mask, image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels == 1:
        return image
    else:
        image_coord_dst_0, image_coord_dst_1 = get_bottom_coordinates(mask)
        for i in range(len(image_coord_dst_0)):
            cv2.circle(image, (image_coord_dst_1[i], image_coord_dst_0[i]), 25, (255, 0, 0), 2)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image_bgr


# FastAPI endpoint
@app.post("/predict_landmarks/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    predicted_mask = predict_image(image_np, model)
    reverted_mask = revert(predicted_mask)
    landmarks = detect(reverted_mask, image_np)
    _, buffer = cv2.imencode('.png', landmarks)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000, timeout_keep_alive=300, timeout_graceful_shutdown=300)
