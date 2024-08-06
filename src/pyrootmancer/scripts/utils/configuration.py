import logging
import os
import os
from colorama import init, Fore, Style

logging.basicConfig(level=logging.INFO)


init(autoreset=True)


def get_project_root(root="2023-24d-fai2-adsai-group-cv5"):
    return os.path.join(os.sep.join(os.getcwd().split(os.sep)[: os.getcwd().split(os.sep).index(root) + 1]))


base_folder = get_project_root()

folder_config = {
    # Data folders - raw, processed and external
    "external_data_folder": os.path.join(base_folder, "data", "external"),
    "raw_data_folder": os.path.join(base_folder, "data", "raw"),
    "processed_data_folder": os.path.join(base_folder, "data", "processed"),
    # Raw data - unpatched - raw images and masks
    "data_unpatched": os.path.join(base_folder, "data", "raw", "data_unpatched"),
    "images_folder_unpatched": os.path.join(base_folder, "data", "raw", "data_unpatched", "images"),
    "test_folder": os.path.join(base_folder, "data", "raw", "data_unpatched", "test"),
    "root_folder_unpatched": os.path.join(base_folder, "data", "raw", "data_unpatched", "root_masks"),
    "shoot_folder_unpatched": os.path.join(base_folder, "data", "raw", "data_unpatched", "shoot_masks"),
    # Raw data - patched - patches of the images and masks
    # Patches of the images and masks for use with ImageDataGenerator()
    "data_patched": os.path.join(base_folder, "data", "raw", "data_patched"),
    "images_folder_patched": os.path.join(base_folder, "data", "raw", "data_patched", "images", "images"),
    "root_folder_patched": os.path.join(base_folder, "data", "raw", "data_patched", "root_masks", "root_masks"),
    "shoot_folder_patched": os.path.join(base_folder, "data", "raw", "data_patched", "shoot_masks", "shoot_masks"),
    # Processed predictions - predicted masks from the raw/data_unpatched/test_folder
    "data_predictions_raw": os.path.join(base_folder, "data", "processed", "data_predictions_raw"),
    "data_predictions_clean": os.path.join(base_folder, "data", "processed", "data_predictions_clean"),
    # Feature outputs:
    "root_labels": os.path.join(base_folder, "data", "external", "root_labels"),
    "root_landmarks": os.path.join(base_folder, "data", "external", "root_landmarks"),
    # Models folder
    "models_folder": os.path.join(base_folder, "models"),
}

param_config = {"patch_size": 256, "input_shape": (256, 256, 3), "num_classes": 1, "batch_size": 16, "epochs": 1}


def print_folders():
    """
    Prints the folder paths specified in the configuration in a colorful way.
    """
    # logging.info(f"{Fore.RED}FOLDER STRUCTURE HELPER")
    for key, value in folder_config.items():
        logging.info(f"{Fore.GREEN}{key}: {Fore.BLUE}{value}")


if __name__ == "__main__":
    print(base_folder)
    print_folders()
