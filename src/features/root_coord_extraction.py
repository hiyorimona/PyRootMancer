import logging
from typing import List, Tuple

import cv2
import numpy as np
from skan import Skeleton, summarize

from src.features.instance_segmentation import InstanceSegmentation
from src.utils.configuration import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LandmarkExtraction:
    """
    A class for extracting landmarks from root images, including removing small components,
    getting bottom coordinates, and displaying root landmarks.

    Author: Thomas Pichardo
    """

    def remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Removes small connected components from the mask based on the number of labels.

        This function filters out small connected components in the input binary mask
        based on predefined size thresholds, which vary according to the number of labels.

        Parameters:
        -----------
        mask : np.ndarray
            The input binary mask.

        Returns:
        --------
        np.ndarray
            The output mask with small components removed.
        """
        try:
            # Get connected components and their stats
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

            # Get sizes of components (excluding the background)
            sizes = stats[1:, -1]

            # Initialize result image with zeros
            im_result = np.zeros_like(labels)

            # Determine the min_size based on the number of labels
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

            # Filter out small components
            for label in range(1, num_labels):
                if sizes[label - 1] >= min_size:
                    im_result[labels == label] = 255

            return im_result.astype(np.uint8)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return np.zeros_like(mask)

    def get_bottom_coordinates(
        self, input_folder: str, num_img: int, threshold: int = 50
    ) -> Tuple[List[int], List[int]]:
        """
        Processes and filters coordinates from the skeleton image.

        This function reads a mask image, removes small components, performs morphological operations,
        extracts skeleton features, filters coordinates based on specific criteria, and sorts them.

        Parameters:
        -----------
        input_folder : str
            Path to the input folder containing the images.
        num_img : int
            Image number to process.
        threshold : int
            Threshold for filtering based on coordinate differences. Default is 50.

        Returns:
        --------
        Tuple[List[int], List[int]]
            Filtered lists of image coordinates.
        """
        try:
            # Initialize instance segmentation for further processing
            segmentation = InstanceSegmentation()
            # List all predicted clean paths
            predicted_clean_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]

            # Read and process the mask
            mask = cv2.imread(predicted_clean_paths[num_img], 0)
            mask = segmentation.opening_closing(self.remove_small_components(mask))

            # Summarize the skeleton
            df = summarize(Skeleton(mask))
            filtered_df = df[
                (df['image-coord-dst-0'] < 2415) & (df['branch-type'] != 0) & (df['euclidean-distance'] > 20)
            ]

            # Extract and sort coordinates
            image_coord_dst_0, image_coord_dst_1 = (
                filtered_df.sort_values(by='image-coord-dst-0', ascending=False)
                .groupby('skeleton-id', as_index=False)
                .agg({'image-coord-dst-0': 'max', 'image-coord-dst-1': 'first'})[
                    ['image-coord-dst-0', 'image-coord-dst-1']
                ]
                .values.T.tolist()
            )

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
                    if abs(sorted_image_coord_dst_1[i] - sorted_image_coord_dst_1[j]) <= threshold:
                        if sorted_image_coord_dst_0[i] < sorted_image_coord_dst_0[j]:
                            indices_to_remove.add(i)
                        else:
                            indices_to_remove.add(j)

            # Create new lists excluding the indices to remove
            filtered_image_coord_dst_0 = [
                val for idx, val in enumerate(sorted_image_coord_dst_0) if idx not in indices_to_remove
            ]
            filtered_image_coord_dst_1 = [
                val for idx, val in enumerate(sorted_image_coord_dst_1) if idx not in indices_to_remove
            ]

            # Pair and sort filtered coordinates
            paired_coords = list(zip(filtered_image_coord_dst_0[:5], filtered_image_coord_dst_1[:5]))
            paired_coords.sort(key=lambda pair: pair[1])
            sorted_image_coord_dst_0, sorted_image_coord_dst_1 = zip(*paired_coords)

            return sorted_image_coord_dst_0, sorted_image_coord_dst_1

        except Exception as e:
            logging.error(f"No roots detected: {e}")
            return [], []

    def display_root_landmarks(self, input_folder: str, test_folder: str, num_img: int) -> None:
        """
        Displays root landmarks on the original image.

        This function reads the original image and its corresponding mask, extracts and draws root landmarks,
        and displays the image with the landmarks highlighted.

        Parameters:
        -----------
        input_folder : str
            Path to the input folder containing the predicted masks.
        test_folder : str
            Path to the test folder containing the original images.
        num_img : int
            Image number to process.

        Returns:
        --------
        None
        """
        # Get paths to the predicted clean masks and original images
        predicted_clean_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
        original_image_paths = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]

        # Read the original image and the corresponding mask
        image = cv2.imread(original_image_paths[num_img])
        mask = cv2.imread(predicted_clean_paths[num_img], 0)

        # Get connected components from the mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        if num_labels == 1:
            # If only one component, display the original image resized
            img_resized = cv2.resize(image, (image.shape[1] // 5, image.shape[0] // 5))
            cv2.imshow('Image', 255 - img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Get bottom coordinates of the root landmarks
            image_coord_dst_0, image_coord_dst_1 = self.get_bottom_coordinates(input_folder, num_img)

            # Draw colored circles on the image for each coordinate
            for i in range(len(image_coord_dst_0)):
                cv2.circle(image, (image_coord_dst_1[i], image_coord_dst_0[i]), 25, (255, 255, 0), 2)  # Yellow circles

            # Convert the image to BGR for displaying
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Display the image
            img_resized = cv2.resize(image_bgr, (image_bgr.shape[1] // 5, image_bgr.shape[0] // 5))
            cv2.imshow('Image', 255 - img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    landmarks = LandmarkExtraction()
    image_num = 1  # Define the image number to process
    landmarks.display_root_landmarks(
        folder_config.get("data_predictions_clean"), folder_config.get("test_folder"), image_num
    )

    y, x = landmarks.get_bottom_coordinates(folder_config.get("data_predictions_clean"), image_num)
    logging.info(f"coordinates x: {x}" f" \ncoordinates y: {y}")
