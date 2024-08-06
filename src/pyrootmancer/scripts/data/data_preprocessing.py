import logging
import shutil
from zipfile import ZipFile

import cv2
import tqdm as tq
from patchify import patchify

from pyrootmancer.scripts.utils.configuration import *

logging.basicConfig(level=logging.INFO)


class DataPipelineSetup:

    def create_folders(self):
        """
        Creates all the directories specified in the configuration.

        This method iterates through the configuration dictionary and creates each directory if it does not already exist.
        """

        for key, value in folder_config.items():
            os.makedirs(value, exist_ok=True)
        print_folders()
        if not os.path.exists(os.path.join(folder_config.get("raw_data_folder"), "train.zip")):
            shutil.move(os.path.join(base_folder, "train.zip"), folder_config.get("raw_data_folder"))
        if not os.path.exists(os.path.join(folder_config.get("raw_data_folder"), "masks.zip")):
            shutil.move(os.path.join(base_folder, "masks.zip"), folder_config.get("raw_data_folder"))
        if not os.path.exists(os.path.join(folder_config.get("raw_data_folder"), "test.zip")):
            shutil.move(os.path.join(base_folder, "test.zip"), folder_config.get("raw_data_folder"))

    def unzip(self, keyword=None):
        """
        Unzips a specified archive and organizes the contents based on the keyword.

        This method extracts the contents of a zip file specified by the `keyword`. It organizes
        the extracted files into appropriate directories based on the `keyword`.
        """
        try:
            # Define the file name and extraction folder
            missing_files_train_str = (
                '000',
                '008',
                '019',
                '023',
                '030',
                '031',
                '032',
                '033',
                '034',
                '035',
                '036',
                '038',
                '039',
                '040',
            )

            file_name = os.path.join(folder_config.get("raw_data_folder"), f"{keyword}.zip")

            extraction_folder = f"{keyword}"

            # Check if there are existing files in the destination folders
            destination_folders = {
                "masks": [folder_config.get("root_folder_unpatched"), folder_config.get("shoot_folder_unpatched")],
                "train": [folder_config.get("images_folder_unpatched")],
                "test": [folder_config.get("test_folder")],
            }

            for folder in destination_folders.get(keyword, []):
                if any(os.scandir(folder)):
                    logging.warning(
                        f"Files already exist in {os.path.basename(folder)}. Aborting unzip operation to prevent overwriting."
                    )
                    return

            # Extract the zip file
            with ZipFile(file_name, 'r') as zip:
                logging.info(f'{keyword} file unzipping...')
                zip.extractall(extraction_folder)

            # Initialize counters
            images = 0
            test_images = 0
            root_masks = 0
            shoot_masks = 0

            # Organize the extracted files
            if keyword == "masks":
                for root, dirs, files in os.walk(extraction_folder):
                    for file in files:
                        if file.lower().endswith(('.tiff', '.tif')):
                            if 'shoot_mask' in file.lower():
                                if file.lower().startswith(missing_files_train_str):
                                    continue
                                destination_folder = folder_config.get("shoot_folder_unpatched")
                                image_path = os.path.join(os.getcwd(), root, file)
                                shoot_masks += 1
                                os.rename(image_path, os.path.join(destination_folder, file))
                            elif 'occluded_root_mask' in file.lower():
                                continue
                            elif 'root_mask' in file.lower():
                                if file.lower().startswith(missing_files_train_str):
                                    continue
                                destination_folder = folder_config.get("root_folder_unpatched")
                                image_path = os.path.join(os.getcwd(), root, file)
                                root_masks += 1
                                os.rename(image_path, os.path.join(destination_folder, file))

                logging.info(f'{root_masks} tiff images class root in {extraction_folder} zip folder')
                logging.info(f'{shoot_masks} tiff images class shoot in {extraction_folder} zip folder')
                logging.info('Done!')

            elif keyword == "train":
                for root, dirs, files in os.walk(extraction_folder):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            if file.lower().startswith(missing_files_train_str):
                                continue
                            destination_folder = folder_config.get("images_folder_unpatched")
                            image_path = os.path.join(os.getcwd(), root, file)
                            images += 1
                            os.rename(image_path, os.path.join(destination_folder, file))

                logging.info(f'{images} png images in {extraction_folder} zip folder')
                logging.info('Done!')

            elif keyword == "test":
                for root, dirs, files in os.walk(extraction_folder):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(os.getcwd(), root, file)
                            test_images += 1
                            os.rename(image_path, os.path.join(folder_config.get("test_folder"), file))

                logging.info(f'{test_images} png images in {extraction_folder} zip folder')
                logging.info('Done!')

                shutil.rmtree(extraction_folder)

        except Exception as e:
            logging.error(f"An error occurred during the unzip operation: {e}")

    def crop(self, folder):
        """
        Crop images in the specified folder.

        This method crops all `.tif` and `.png` images in the given folder.
        The cropped area is defined by specific pixel ranges. For `.png` images,
        the cropped area is normalized before being saved.
        """
        try:
            image_files = [file for file in os.listdir(folder) if file.lower().endswith(('.tif', '.png'))]
            loop = tq.tqdm(
                enumerate(image_files),
                total=len(image_files),
                bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[38;2;70;130;180m', '\033[0m'),
            )

            for _, file in loop:
                image_path = os.path.join(folder, file)
                try:
                    if image_path.lower().endswith('png'):
                        image = cv2.imread(image_path)
                        if image is not None:
                            if image.shape[:2] != (2731, 2752):
                                normalized_image = cv2.normalize(
                                    image[75 : image.shape[0] - 200, 750 : image.shape[1] - 700],
                                    None,
                                    0,
                                    255,
                                    cv2.NORM_MINMAX,
                                )
                                cv2.imwrite(image_path, normalized_image)
                        else:
                            logging.error(f"Failed to read PNG image: {image_path}")

                    elif image_path.lower().endswith('tif'):
                        mask = cv2.imread(image_path, 0)
                        if mask is not None:
                            if mask.shape != (2731, 2752):
                                cv2.imwrite(image_path, mask[75 : mask.shape[0] - 200, 750 : mask.shape[1] - 700])
                        else:
                            logging.error(f"Failed to read TIFF image: {image_path}")
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")

            logging.info(f"Cropping completed successfully from {os.path.basename(folder)}")

        except Exception as e:
            logging.error(f"An error occurred while cropping images in folder {os.path.basename(folder)}: {e}")

    def padder(self, image):
        """
        Adds padding to an image to make its dimensions divisible by a specified patch size.

        This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to black (0, 0, 0).

        Parameters:
            - image (numpy.ndarray): The input image as a NumPy array. Expected shape is (height, width, channels).
            - patch_size (int): The patch size to which the image dimensions should be divisible. It's applied to both height and width.

        Returns:
            - numpy.ndarray: The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be divisible by the specified patch size.

        """
        patch_size = param_config.get("patch_size")

        h = image.shape[0]
        w = image.shape[1]
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

    def img_patchify(self, img_dir, save_dir):
        """
        Adds padding to all images in folder and patchifies them using the patchify library.
        The patchified images get saved in the specified folder and returned in an array.

        Args:
            img_dir (str): Path to image folder
            save_dir (str): Path to folder in which the patches should be saved
            patch_size (int): Size of patches that should be made

        Returns:
            array: Array of patched images
        """
        try:
            img = []
            tifs = []
            patch_size = param_config.get("patch_size")

            image_files = [file for file in os.listdir(img_dir) if file.lower().endswith(('.tif', '.png'))]
            loop = tq.tqdm(
                enumerate(image_files),
                total=len(image_files),
                bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[38;2;70;130;180m', '\033[0m'),
            )

            for _, image_filename in loop:
                if image_filename.endswith((".png", ".tif")):
                    image_path = os.path.join(img_dir, image_filename)
                    im = cv2.imread(image_path)
                    img_name, extension = os.path.splitext(image_filename)

                    padded_image = self.padder(im)
                    channels = 3 if extension == ".png" else 1
                    patches = patchify(padded_image, (patch_size, patch_size, channels), step=patch_size)
                    patches = patches.reshape(-1, patch_size, patch_size, channels)

                    for i, patch in enumerate(patches):
                        output_filename = f"{img_name}{i}{extension}"
                        cv2.imwrite(os.path.join(save_dir, output_filename), patch)

                        if extension == ".png":
                            img.append(patch)
                        elif extension == ".tif":
                            tifs.append(patch)
            return img, tifs

        except Exception as e:
            logging.error(f"Error processing {os.path.basename(img_dir)}: {e}")
        finally:
            logging.info(f"Patches for {os.path.basename(img_dir)} created and stored successfully!")


if __name__ == "__main__":
    processor = DataPipelineSetup()

    processor.create_folders()

    processor.unzip("train")
    processor.unzip("test")
    processor.unzip("masks")

    processor.crop(folder_config.get("images_folder_unpatched"))
    processor.crop(folder_config.get("root_folder_unpatched"))
    processor.crop(folder_config.get("shoot_folder_unpatched"))

    processor.img_patchify(folder_config.get("images_folder_unpatched"), folder_config.get("images_folder_patched"))
    processor.img_patchify(folder_config.get("root_folder_unpatched"), folder_config.get("root_folder_patched"))
    processor.img_patchify(folder_config.get("shoot_folder_unpatched"), folder_config.get("shoot_folder_patched"))
