import logging

import cv2
import numpy as np
import tqdm as tq
from skimage.transform import resize

from prc.data.data_preprocessing import DataPipelineSetup
from prc.models.model_training import ModelTraining
from prc.utils.configuration import *

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


class InstanceSegmentation:
    """
    A class for performing instance segmentation tasks including noise removal, overlaying masks,
    and resizing predicted masks to match the size of original images.

    @author: Cédric Verhaegh
    """

    def opening_closing(self, img: np.array) -> np.array:
        """
        Applies morphological operations to remove noise from the input image.
        Performs erosion, dilation, and closing operations to clean the image.

        Parameters:
        -----------
        img : np.array
            The input image to be processed.

        Returns:
        --------
        np.array
            The processed image.
        """
        # Create a kernel for morphological operations
        kernel = np.ones((6, 6), dtype="uint8")
        # Apply erosion to the image
        im_erosion = cv2.erode(img, kernel, iterations=1)
        # Apply dilation to the eroded image
        im_dilation = cv2.dilate(im_erosion, kernel, iterations=2)
        # Apply erosion to the dilated image (closing operation)
        im_closing = cv2.erode(im_dilation, kernel, iterations=1)

        return im_closing

    def test_overlaying(
        self,
        image_path: str,
        output_folder: str,
        model_folder: str,
        model_name: str,
    ) -> np.array:
        """
        Overlays a predicted mask on a randomly selected original image and displays the result.

        This function selects a random image and its corresponding predicted mask from predefined
        directories, blends them to highlight the mask in red on the original image, and shows the
        blended image.

        Parameters:
        -----------
        image_path : str
            The path to the input image.
        output_folder : str
            The path to the folder where the output will be saved.
        model_folder : str
            The path to the folder containing the model.
        model_name : str
            The name of the model to be used for prediction.

        Returns:
        --------
        np.array
            The blended image with the overlayed mask.
        """
        # Initialize model training
        modelling = ModelTraining()
        # Predict the mask for the image
        predicted_mask = modelling.predict_image(
            image_path, output_folder, model_folder, model_name
        )

        # Read the test image in grayscale
        test_image = cv2.imread(image_path, 0)
        # Crop the predicted mask to match the region of interest (ROI) in the test image
        im = predicted_mask[0:2816, 0:2816]
        roi_1 = test_image[50: 2816 + 50, 720: 2816 + 720]

        # Resize the predicted mask to fit the ROI
        overlay_image = (
            resize(im, roi_1.shape, mode='reflect', anti_aliasing=True) * 255
        ).astype(np.uint8)
        # Create a blank image with the same shape as the test image
        modified_cropped = np.zeros_like(test_image)

        # Define the ROI in the blank image
        roi = modified_cropped[50: 2816 + 50, 720: 2816 + 720]
        # Overlay the predicted mask onto the ROI
        result = cv2.addWeighted(roi, 1, overlay_image, 0.7, 0)

        # Place the result back into the blank image
        modified_cropped[50: 2816 + 50, 720: 2816 + 720] = result
        # Normalize the modified image
        norm = cv2.normalize(modified_cropped, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the test image to BGR color space
        base_img_colored = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
        # Create a blank image for the overlay in red channel
        overlay_img_red = np.zeros_like(base_img_colored)
        overlay_img_red[:, :, 2] = norm

        # Blend the base image and the overlay image
        blended_img = cv2.addWeighted(
            base_img_colored, 0.45, overlay_img_red, 1 - 0.45, 0
        )
        # Resize the blended image for display
        img_resized = cv2.resize(
            blended_img, (blended_img.shape[1] // 5, blended_img.shape[0] // 5)
        )

        # Display the final image
        cv2.imshow('Image', 255 - img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return blended_img

    def return_original_size_image(
        self, image_path: str, output_folder: str
    ) -> np.array:
        """
        Resizes predicted masks to match the size of the original images and overlays them.

        This function reads each pair of predicted mask and original image, crops and resizes
        the predicted mask to fit the corresponding region in the original image, overlays the
        resized mask onto the original image, and saves the result. This process ensures the
        predicted mask aligns correctly with the original image for further analysis or display.

        Parameters:
        -----------
        image_path : str
            The path to the predicted mask image.
        output_folder : str
            The path to the folder where the resized image will be saved.

        Returns:
        --------
        np.array
            The resized image with the overlayed mask.
        """
        # Create a blank original image with predefined dimensions
        original_image = np.zeros((3006, 4202), dtype=np.uint8)
        # Read the predicted mask image in grayscale
        predicted_mask = cv2.imread(image_path, 0)
        # Crop the predicted mask to match the region of interest (ROI)
        im = predicted_mask[0:2816, 0:2816]
        roi_1 = original_image[50: 2816 + 50, 720: 2816 + 720]

        # Resize the predicted mask to fit the ROI
        overlay_image = (
            resize(im, roi_1.shape, mode='reflect', anti_aliasing=True) * 255
        ).astype(np.uint8)
        # Create a blank image with the same shape as the original image
        modified_cropped = np.zeros_like(original_image)

        # Define the ROI in the blank image
        roi = modified_cropped[50: 2816 + 50, 720: 2816 + 720]
        # Overlay the predicted mask onto the ROI
        result = cv2.addWeighted(roi, 1, overlay_image, 0.7, 0)

        # Place the result back into the blank image
        modified_cropped[50: 2816 + 50, 720: 2816 + 720] = result
        # Normalize the modified image
        norm = cv2.normalize(modified_cropped, None, 0, 255, cv2.NORM_MINMAX)

        # Save the final normalized image to the output folder
        cv2.imwrite(
            os.path.join(output_folder, os.path.basename(image_path)), norm
        )
        return norm

    def return_original_size_folder(
        self, test_folder: str, output_folder: str
    ) -> None:
        """
        Resizes predicted masks to match the size of the original images and overlays them for all images in a folder.

        This function reads each pair of predicted mask and original image from the test folder,
        crops and resizes the predicted mask to fit the corresponding region in the original image,
        overlays the resized mask onto the original image, and saves the result. This process ensures the
        predicted mask aligns correctly with the original image for further analysis or display.

        Parameters:
        -----------
        test_folder : str
            The path to the folder containing test images.
        output_folder : str
            The path to the folder where the resized images will be saved.

        Returns:
        --------
        None
        """
        # Initialize data pipeline setup
        processor = DataPipelineSetup()
        # Create necessary folders for processing
        processor.create_folders()

        # Get the list of test image paths
        test_folder_path = [
            os.path.join(test_folder, file) for file in os.listdir(test_folder)
        ]
        # Create a tqdm loop for progress tracking
        loop = tq.tqdm(
            enumerate(test_folder_path),
            total=len(test_folder_path),
            bar_format='{l_bar}%s{bar}%s{r_bar}'
            % ('\033[38;2;70;130;180m', '\033[0m'),
        )

        # Process each test image
        for _, test_path in loop:
            _ = self.return_original_size_image(test_path, output_folder)

        logging.info("Done!")
