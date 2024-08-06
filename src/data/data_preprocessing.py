import logging
import os
import shutil
from zipfile import ZipFile

import cv2
import tqdm as tq
from patchify import patchify
import numpy as np

from src.utils.configuration import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPipelineSetup:
    """
    A class to set up and preprocess data for the project, including folder creation,
    unzipping archives, cropping images, adding padding, and patchifying images.

    Author: Simona Dimitrova
    """

    def create_folders(self) -> None:
        """
        Creates all the directories specified in the configuration.

        This method iterates through the configuration dictionary and creates each directory if it does not already exist.
        """
        for key, value in folder_config.items():
            os.makedirs(value, exist_ok=True)  # Create the directory if it does not exist

    def unzip(self, keyword: str) -> None:
        """
        Unzips a specified archive and organizes the contents based on the keyword.

        This method extracts the contents of a zip file specified by the `keyword`. It organizes
        the extracted files into appropriate directories based on the `keyword`.

        Parameters:
        -----------
        keyword : str
            The keyword to identify which zip file to extract and how to organize its contents.
        """
        try:
            # Define the file name and extraction folder based on the keyword
            missing_files_train_str = (
                '000', '008', '019', '023', '030', '031', '032', '033', '034', '035', '036', '038', '039', '040'
            )

            file_name = os.path.join(folder_config.get("raw_data_folder"), f"{keyword}.zip")
            extraction_folder = f"{keyword}"

            # Define the destination folders for each keyword
            destination_folders = {
                "masks": [
                    folder_config.get("root_folder_unpatched"),
                    folder_config.get("shoot_folder_unpatched"),
                ],
                "train": [folder_config.get("images_folder_unpatched")],
                "test": [folder_config.get("test_folder")],
            }

            # Check if files already exist in the destination folders to avoid overwriting
            for folder in destination_folders.get(keyword, []):
                if any(os.scandir(folder)):
                    logging.warning(
                        f"Files already exist in {os.path.basename(folder)}. Aborting unzip operation to prevent overwriting."
                    )
                    return

            # Extract the zip file
            with ZipFile(file_name, 'r') as zip_ref:
                logging.info(f'{keyword} file unzipping...')
                zip_ref.extractall(extraction_folder)

            # Initialize counters for different file types
            images, test_images, root_masks, shoot_masks = 0, 0, 0, 0

            # Organize the extracted files based on the keyword
            if keyword == "masks":
                for root, dirs, files in os.walk(extraction_folder):
                    for file in files:
                        if file.lower().endswith(('.tiff', '.tif')):
                            if 'shoot_mask' in file.lower():
                                if file.lower().startswith(
                                    missing_files_train_str
                                ):
                                    continue
                                destination_folder = folder_config.get(
                                    "shoot_folder_unpatched"
                                )
                                image_path = os.path.join(
                                    os.getcwd(), root, file
                                )
                                shoot_masks += 1
                                os.rename(
                                    image_path,
                                    os.path.join(destination_folder, file),
                                )
                            elif 'occluded_root_mask' in file.lower():
                                continue
                            elif 'root_mask' in file.lower():
                                if file.lower().startswith(
                                    missing_files_train_str
                                ):
                                    continue
                                destination_folder = folder_config.get(
                                    "root_folder_unpatched"
                                )
                                image_path = os.path.join(
                                    os.getcwd(), root, file
                                )
                                root_masks += 1
                                os.rename(
                                    image_path,
                                    os.path.join(destination_folder, file),
                                )

                logging.info(
                    f'{root_masks} tiff images class root in {extraction_folder} zip folder'
                )
                logging.info(
                    f'{shoot_masks} tiff images class shoot in {extraction_folder} zip folder'
                )
                logging.info('Done!')

            elif keyword == "train":
                for root, dirs, files in os.walk(extraction_folder):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            if file.lower().startswith(
                                missing_files_train_str
                            ):
                                continue
                            destination_folder = folder_config.get(
                                "images_folder_unpatched"
                            )
                            image_path = os.path.join(os.getcwd(), root, file)
                            images += 1
                            os.rename(
                                image_path,
                                os.path.join(destination_folder, file),
                            )

                logging.info(
                    f'{images} png images in {extraction_folder} zip folder'
                )
                logging.info('Done!')

            elif keyword == "test":
                for root, dirs, files in os.walk(extraction_folder):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(os.getcwd(), root, file)
                            test_images += 1
                            os.rename(
                                image_path,
                                os.path.join(
                                    folder_config.get("test_folder"), file
                                ),
                            )

                logging.info(
                    f'{test_images} png images in {extraction_folder} zip folder'
                )
                logging.info('Done!')

                shutil.rmtree(extraction_folder)  # Remove the extraction folder after processing

        except Exception as e:
            logging.error(f"An error occurred during the unzip operation: {e}")

    def crop(self, folder: str) -> None:
        """
        Crops images in the specified folder.

        This method crops all `.tif` and `.png` images in the given folder.
        The cropped area is defined by specific pixel ranges. For `.png` images,
        the cropped area is normalized before being saved.

        Parameters:
        -----------
        folder : str
            The folder containing images to be cropped.
        """
        try:
            # Get a list of image files in the folder
            image_files = [file for file in os.listdir(folder) if file.lower().endswith(('.tif', '.png'))]
            loop = tq.tqdm(enumerate(image_files), total=len(image_files),
                           bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[38;2;70;130;180m', '\033[0m'))

            for _, file in loop:
                image_path = os.path.join(folder, file)
                try:
                    if image_path.lower().endswith('png'):
                        # Read and normalize the image if it's a PNG file
                        image = cv2.imread(image_path)
                        if image is not None:
                            if image.shape[:2] != (2731, 2752):
                                normalized_image = cv2.normalize(
                                    image[
                                        75: image.shape[0] - 200,
                                        750: image.shape[1] - 700,
                                    ],
                                    None,
                                    0,
                                    255,
                                    cv2.NORM_MINMAX,
                                )
                                cv2.imwrite(image_path, normalized_image)
                        else:
                            logging.error(
                                f"Failed to read PNG image: {image_path}"
                            )

                    elif image_path.lower().endswith('tif'):
                        # Read and crop the image if it's a TIFF file
                        mask = cv2.imread(image_path, 0)
                        if mask is not None:
                            if mask.shape != (2731, 2752):
                                cv2.imwrite(
                                    image_path,
                                    mask[
                                        75: mask.shape[0] - 200,
                                        750: mask.shape[1] - 700,
                                    ],
                                )
                        else:
                            logging.error(
                                f"Failed to read TIFF image: {image_path}"
                            )
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")

            logging.info(
                f"Cropping completed successfully from {os.path.basename(folder)}"
            )

        except Exception as e:
            logging.error(
                f"An error occurred while cropping images in folder {os.path.basename(folder)}: {e}"
            )

    def padder(self, image: np.ndarray) -> np.ndarray:
        """
        Adds padding to an image to make its dimensions divisible by a specified patch size.

        This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to black (0, 0, 0).

        Parameters:
        -----------
        image : np.ndarray
            The input image as a NumPy array. Expected shape is (height, width, channels).

        Returns:
        --------
        np.ndarray
            The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be divisible by the specified patch size.
        """
        patch_size = param_config.get("patch_size")  # Get the patch size from the configuration

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

    def img_patchify(self, img_dir: str, save_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Adds padding to all images in a folder and patchifies them using the patchify library.
        The patchified images are saved in the specified folder and returned in an array.

        Parameters:
        -----------
        img_dir : str
            Path to the image folder.
        save_dir : str
            Path to the folder in which the patches should be saved.

        Returns:
        --------
        tuple[list[np.ndarray], list[np.ndarray]]
            Arrays of patched images.
        """
        try:
            img, tifs = [], []
            patch_size = param_config.get("patch_size")  # Get the patch size from the configuration

            # Get a list of image files in the directory
            image_files = [file for file in os.listdir(img_dir) if file.lower().endswith(('.tif', '.png'))]
            loop = tq.tqdm(enumerate(image_files), total=len(image_files),
                           bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[38;2;70;130;180m', '\033[0m'))

            for _, image_filename in loop:
                if image_filename.endswith((".png", ".tif")):
                    image_path = os.path.join(img_dir, image_filename)
                    im = cv2.imread(image_path)  # Read the image
                    img_name, extension = os.path.splitext(image_filename)

                    padded_image = self.padder(im)  # Add padding to the image
                    channels = 3 if extension == ".png" else 1  # Determine the number of channels
                    patches = patchify(padded_image, (patch_size, patch_size, channels), step=patch_size)
                    patches = patches.reshape(-1, patch_size, patch_size, channels)  # Reshape the patches

                    for i, patch in enumerate(patches):
                        output_filename = f"{img_name}_{i}{extension}"
                        cv2.imwrite(os.path.join(save_dir, output_filename), patch)  # Save each patch

                        if extension == ".png":
                            img.append(patch)  # Add to the list of PNG patches
                        elif extension == ".tif":
                            tifs.append(patch)  # Add to the list of TIFF patches

            return img, tifs

        except Exception as e:
            logging.error(f"Error processing {os.path.basename(img_dir)}: {e}")
        finally:
            logging.info(
                f"Patches for {os.path.basename(img_dir)} created and stored successfully!"
            )


if __name__ == "__main__":
    processor = DataPipelineSetup()

    processor.create_folders()  # Create necessary folders

    processor.unzip("train")  # Unzip and organize training data
    processor.unzip("test")  # Unzip and organize test data
    processor.unzip("masks")  # Unzip and organize mask data

    processor.crop(folder_config.get("images_folder_unpatched"))  # Crop images in the unpatched images folder
    processor.crop(folder_config.get("root_folder_unpatched"))  # Crop images in the unpatched root masks folder
    processor.crop(folder_config.get("shoot_folder_unpatched"))  # Crop images in the unpatched shoot masks folder

    processor.img_patchify(folder_config.get("images_folder_unpatched"), folder_config.get(
        "images_folder_patched"))  # Patchify images in the unpatched images folder
    processor.img_patchify(folder_config.get("root_folder_unpatched"), folder_config.get(
        "root_folder_patched"))  # Patchify images in the unpatched root masks folder
    processor.img_patchify(folder_config.get("shoot_folder_unpatched"), folder_config.get(
        "shoot_folder_patched"))  # Patchify images in the unpatched shoot masks folder
