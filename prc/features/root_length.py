import os
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

from prc.models.model_evaluation import f1, iou

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RootLengthCalculator:
    """
    A class to calculate root lengths from images using a trained model. The class supports
    prediction of masks, analysis of images to extract root lengths, and saving results.

    Author: Samuel Vieira Vasconcelos
    """

    def __init__(self, img_dir: str, model_path: str, custom_objects: dict):
        """
        Initializes the RootLengthCalculator class.

        Parameters:
        -----------
        img_dir : str
            Path to the directory containing the images.
        model_path : str
            Path to the trained model.
        custom_objects : dict
            Custom objects needed for the model.
        """
        self.img_dir = img_dir
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        self.results = []
        self.part_bounding_boxes = {}

    def predict_image(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """
        Predicts the mask for the given image using the loaded model.

        This function resizes the input image, normalizes it, and predicts the mask using
        the loaded model. The predicted mask is then resized back to the original image size.

        Parameters:
        -----------
        img : np.ndarray
            The image to predict.
        target_size : int
            The target size for the model input.

        Returns:
        --------
        np.ndarray
            The predicted mask.
        """
        img_resized = cv2.resize(img, (target_size, target_size))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0
        predicted_mask = self.model.predict(img_array)[0]
        predicted_mask_resized = cv2.resize(predicted_mask, (img.shape[1], img.shape[0]))
        return predicted_mask_resized

    def process_images(self) -> None:
        """
        Processes all images in the specified directory.

        This function iterates over all images in the directory, predicts masks,
        and analyzes each image to extract root lengths.
        """
        folder_path = self.img_dir

        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.endswith(".tif"):
                file_path = os.path.join(folder_path, file_name)
                img = cv2.imread(file_path)
                predicted_mask = self.predict_image(img, 256)
                self.analyze_image(file_name, img, predicted_mask)

    def analyze_image(
        self, file_name: str, img: np.ndarray, predicted_mask: np.ndarray
    ) -> None:
        """
        Analyzes the image to find and annotate parts based on the predicted mask.

        This function processes the predicted mask to identify connected components,
        filters these components, and calculates the root length for each part of the plant.

        Parameters:
        -----------
        file_name : str
            The name of the image file.
        img : np.ndarray
            The original image.
        predicted_mask : np.ndarray
            The predicted mask for the image.
        """
        predicted_mask_uint8 = (predicted_mask > 0.05).astype(np.uint8)
        preds = np.array(predicted_mask_uint8, dtype=np.uint8)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            preds
        )

        # Sort components by area
        sorted_components = sorted(range(1, retval), key=lambda x: stats[x, cv2.CC_STAT_AREA], reverse=True)
        min_area = 200
        min_top = 300
        max_top = 1000
        max_left = 2600

        # Select components based on criteria
        selected_components = [
            i
            for i in sorted_components
            if min_area <= stats[i, cv2.CC_STAT_AREA]
            and min_top <= stats[i, cv2.CC_STAT_TOP] <= max_top
            and stats[i, cv2.CC_STAT_LEFT] <= max_left
        ]

        filtered_image = np.zeros_like(labels, dtype=np.uint8)
        part_boxes = {}
        height, width, _ = img.shape
        part_width = (width // 5) - 30

        if not selected_components:
            for part_number in range(1, 6):
                self.results.append(
                    {
                        "Plant ID": f"{os.path.splitext(file_name)[0]}_plant_{part_number}",
                        "Length (px)": 0,
                    }
                )
        else:
            for idx, component_idx in enumerate(selected_components, start=1):
                x, y, w, h = (
                    stats[component_idx, cv2.CC_STAT_LEFT],
                    stats[component_idx, cv2.CC_STAT_TOP],
                    stats[component_idx, cv2.CC_STAT_WIDTH],
                    stats[component_idx, cv2.CC_STAT_HEIGHT],
                )
                part_number = min(x // part_width + 1, 5)

                if part_number not in part_boxes or (
                    w * h
                    > part_boxes[part_number][2] * part_boxes[part_number][3]
                    and x >= (part_number - 1) * part_width
                    and x <= part_number * part_width
                ):
                    part_boxes[part_number] = (x, y, w, h)

                filtered_image[labels == component_idx] = part_number

            for part_number, bbox in part_boxes.items():
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(
                    img,
                    str(part_number),
                    (x + w // 2 - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cropped_roots = [
                filtered_image[y: y + h, x: x + w]
                for x, y, w, h in part_boxes.values()
            ]
            skeletonized_roots = [
                skeletonize(root > 0) for root in cropped_roots
            ]

            for part_number in range(1, 6):
                if (
                    part_number <= len(skeletonized_roots)
                    and part_number in part_boxes
                ):
                    skeleton = skeletonized_roots[part_number - 1]
                    test = summarize(Skeleton(skeleton))
                    length = self.calculate_length(skeleton, test)
                else:
                    length = 0

                file_name_for_part = os.path.splitext(file_name)[0]
                self.results.append(
                    {
                        "Plant ID": f"{file_name_for_part}_plant_{part_number}",
                        "Length (px)": length,
                    }
                )

    def calculate_length(
        self, skeleton: np.ndarray, summary: pd.DataFrame
    ) -> float:
        """
        Calculates the length of the main root.

        This function uses the summarized skeleton data to calculate the total length
        of the main root based on euclidean distances between nodes.

        Parameters:
        -----------
        skeleton : np.ndarray
            The skeletonized root image.
        summary : pd.DataFrame
            The summary from skan.

        Returns:
        --------
        float
            The length of the main root.
        """
        main_max = summary["node-id-dst"].idxmax()
        main_min = summary["node-id-src"].idxmin()
        length = (
            summary.loc[main_max, "euclidean-distance"]
            + summary.loc[main_min, "euclidean-distance"]
        )
        return length

    def save_results(self, csv_filename: str) -> None:
        """
        Saves the results to a CSV file.

        This function converts the results list to a DataFrame and appends it to
        an existing CSV file or creates a new one if it doesn't exist.

        Parameters:
        -----------
        csv_filename : str
            The name of the CSV file.
        """
        df = pd.DataFrame(self.results)
        file_exists = os.path.exists(csv_filename)
        df.to_csv(csv_filename, index=False, mode="a", header=not file_exists)
        logging.info(f"Results saved to {csv_filename}")
