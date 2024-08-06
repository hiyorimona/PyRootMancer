import logging
import argparse
import os
from typing import List, Tuple

import cv2
import tqdm as tq
from patchify import patchify
import numpy as np
from azureml.core import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os


class DataPipelineSetup:
    """
    A class to set up and preprocess data for the project, including folder creation,
    unzipping archives, cropping images, adding padding, and patchifying images.

    Author: Simona Dimitrova
    """
    def test_crop(self, folder, save_dir):

        image_files = [file for file in os.listdir(folder) if file.lower().endswith(('.tif', '.png'))]
        image_path = os.path.join(folder, image_files[3])


        print(f"save dir {save_dir}")
        image = cv2.imread(image_path)
        print(f"image path {image_path}")
        normalized_image = cv2.normalize(image[75:image.shape[0] - 200, 750:image.shape[1] - 700,],None,0,255,cv2.NORM_MINMAX,)
        cv2.imwrite(os.path.join(save_dir,"images", os.path.basename(image_path)), normalized_image)


    def crop_masks(self, folder_masks, save_dir):

        image_files = [file for file in os.listdir(folder_masks) if file.lower().endswith(('.tif'))]
        for file in image_files:
            image_path = os.path.join(folder_masks, file)
            print(f"masks - {os.path.basename(image_path)}")
            mask = cv2.imread(image_path, 0)
            print(f"path to the save mask is - {os.path.join(save_dir, os.path.basename(image_path))}")
            if mask.shape != (2731, 2752):
                cropped_mask = mask[75:mask.shape[0] - 200, 750:mask.shape[1] - 700]
                cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), cropped_mask)
            else:
                cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), mask)



    def crop_images(self, folder_images, save_dir):

        image_files = [file for file in os.listdir(folder_images) if file.lower().endswith(('.png'))]
        for file in image_files:
            image_path = os.path.join(folder_images, file)
            print(f"images - {os.path.basename(image_path)}")
            image = cv2.imread(image_path)
            print(f"path to the save image is - {os.path.join(save_dir, os.path.basename(image_path))}")
            if image.shape[:2] != (2731, 2752):
                normalized_image = cv2.normalize(image[75:image.shape[0] - 200, 750:image.shape[1] - 700,],None,0,255,cv2.NORM_MINMAX,)
                cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), normalized_image)
            else:
                normalized_image = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX,)
                cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), normalized_image)





    def crop(self, folder: str, save_dir: str) -> None:
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
        image_files = [file for file in os.listdir(folder) if file.lower().endswith(('.tif', '.png'))]
        loop = tq.tqdm(enumerate(image_files), total=len(image_files), bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[38;2;70;130;180m', '\033[0m'))

        for _, file in loop:
            image_path = os.path.join(folder, file)
            if image_path.lower().endswith('png'):
                image = cv2.imread(image_path)
                if image is not None:
                    if image.shape[:2] != (2731, 2752):
                        normalized_image = cv2.normalize(image[75 : image.shape[0] - 200, 750 : image.shape[1] - 700,],None,0,255,cv2.NORM_MINMAX,)
                        cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), normalized_image)

            elif image_path.lower().endswith('tif'):
                mask = cv2.imread(image_path, 0)
                if mask is not None:
                    if mask.shape != (2731, 2752):
                        cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), mask[75:mask.shape[0]-200,750:mask.shape[1]-700,])


        logging.info(f"Cropping completed successfully from {os.path.basename(folder)}")



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

    def img_patchify(self, img_dir: str, save_dir: str, type_class: str) -> None:
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
        # save_dir = os.path.join(save_dir, type_class)
        if type_class not in os.listdir(save_dir):
            save_dir = os.path.join(save_dir, type_class)
            os.makedirs(save_dir)

        # if not save_dir:
        #     os.makedirs(save_dir)

        try:
            patch_size = 256 # Get the patch size from the configuration

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
                        print(f"locations to where the files are saved {os.path.join(save_dir, output_filename)}")
                        cv2.imwrite(os.path.join(save_dir, output_filename), patch)  # Save each patch

        except Exception as e:
            logging.error(f"Error processing {os.path.basename(img_dir)}: {e}")
        finally:
            logging.info(
                f"Patches for {os.path.basename(img_dir)} created and stored successfully!")

def main():

    processor = DataPipelineSetup()
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument("function", type=str, help="Function to call: 'crop' or 'img_patchify'")
    parser.add_argument("--images-path", type=str, default="data", help="Path to the images")
    parser.add_argument("--masks-path", type=str, default="data", help="Path to the masks")

    parser.add_argument("--cropped-images-path", type=str, default="data", help="Path to the images")
    parser.add_argument("--cropped-masks-path", type=str, default="data", help="Path to the masks")

    parser.add_argument("--images-patch-path", type=str, default="data", help="Path to the images")
    parser.add_argument("--masks-patch-path", type=str, default="data", help="Path to the masks")

    parser.add_argument("--type-images", type=str, default="images", help="type of the folder")
    parser.add_argument("--type-masks", type=str, default="images", help="type of the folder")
    parser.add_argument("--path", type=str, default="images", help="type of the folder")


    args = parser.parse_args()

    if args.function == "crop-images":
        processor.crop_images(args.images_path, args.cropped_images_path)
    elif args.function == "crop-masks":
        processor.crop_masks(args.masks_path, args.cropped_masks_path)
    elif args.function == "crop":
        processor.crop(args.images_path, args.cropped_masks_path)
        processor.crop(args.masks_path, args.cropped_masks_path)
    elif args.function == "patchify":
        processor.img_patchify(args.cropped_images_path, args.images_patch_path, args.type_images)
        processor.img_patchify(args.cropped_masks_path, args.masks_patch_path, args.type_masks)
    elif args.function == "test-crop":
        processor.test_crop(args.path, args.cropped_images_path)
    else:
        print(f"Unknown function: {args.function}")

#%%

if __name__ == "__main__":
    main()