"""
Azure DataStore - Uploading Data Patches to Azure DataStore

Author: Cédric Verhaegh

This script uploads the processed data patches from the train.zip and masks.zip files to the Azure DataStore.
Before uploading, it splits the data patches (from the training data) into a train and test set, ensuring full images are split correctly.
Then, after splitting the data, it uploads it to the Azure DataStore.
"""

import sys
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import InteractiveLoginAuthentication

# from utils.configuration import *
import os
from tqdm import tqdm
import shutil


class AzureDataUploader:
    def __init__(
        self, subscription_id, resource_group, workspace_name, base_path, auth
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.base_path = base_path
        self.auth = auth
        self.workspace = self.setup_workspace()
        # self.data_patched_path = folder_config.get("data_patched")
        # self.image_dir = folder_config.get("images_folder_patched")
        # self.root_mask_dir = folder_config.get("root_folder_patched")
        self.check_paths()
        self.image_files, self.root_mask_files = self.list_files()

    def setup_workspace(self):
        workspace = Workspace(
            subscription_id=self.subscription_id,
            resource_group=self.resource_group,
            workspace_name=self.workspace_name,
            auth=self.auth,
        )
        return workspace

    def check_paths(self):
        if not os.path.exists(self.data_patched_path):
            raise FileNotFoundError(
                f"The directory {self.data_patched_path} does not exist."
            )
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(
                f"The directory {self.image_dir} does not exist."
            )
        if not os.path.exists(self.root_mask_dir):
            raise FileNotFoundError(
                f"The directory {self.root_mask_dir} does not exist."
            )
        print(f"Data patched directory: {self.data_patched_path}")
        print(f"Image directory: {self.image_dir}")
        print(f"Root mask directory: {self.root_mask_dir}")

    def list_files(self):
        image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        )
        root_mask_files = sorted(
            [f for f in os.listdir(self.root_mask_dir) if f.endswith('.tif')]
        )
        print(
            f"Found {len(image_files)} image patches and {len(root_mask_files)} root mask patches."
        )
        return image_files, root_mask_files

    def prepare_patches(self):
        total_complete_images = len(self.image_files) // 121
        train_ratio = 0.8
        num_train_images = int(round(total_complete_images * train_ratio))

        print(f"Total complete images: {total_complete_images}")
        print(f"Number of training images: {num_train_images}")
        print(
            f"Number of testing images: {total_complete_images - num_train_images}"
        )

        train_images = []
        train_root_masks = []
        test_images = []
        test_root_masks = []

        for i in range(total_complete_images):
            for j in range(121):
                if i < num_train_images:
                    train_images.append(
                        os.path.join(
                            self.image_dir, self.image_files[i * 121 + j]
                        )
                    )
                    train_root_masks.append(
                        os.path.join(
                            self.root_mask_dir,
                            self.root_mask_files[i * 121 + j],
                        )
                    )
                else:
                    test_images.append(
                        os.path.join(
                            self.image_dir, self.image_files[i * 121 + j]
                        )
                    )
                    test_root_masks.append(
                        os.path.join(
                            self.root_mask_dir,
                            self.root_mask_files[i * 121 + j],
                        )
                    )

        self.copy_files(train_images, train_root_masks, "images", "masks")
        self.copy_files(train_root_masks, train_root_masks, "images", "masks")

        # self.copy_files(test_images, test_root_masks, os.path.join(self.data_patched_path, "test/images"), os.path.join(self.data_patched_path, "test/masks"))

        return train_images, train_root_masks

    def copy_files(
        self, image_list, root_mask_list, target_image_dir, target_mask_dir
    ):
        os.makedirs(target_image_dir, exist_ok=True)
        os.makedirs(target_mask_dir, exist_ok=True)
        for img, mask in tqdm(
            zip(image_list, root_mask_list),
            total=len(image_list),
            desc=f"Preparing {target_image_dir}",
        ):
            shutil.copy(img, target_image_dir)
            shutil.copy(mask, target_mask_dir)

    def split_train_validation(self):
        train_image_path = os.path.join(self.data_patched_path, "train/images")
        train_mask_path = os.path.join(self.data_patched_path, "train/masks")

        train_images = sorted(
            [
                os.path.join(train_image_path, f)
                for f in os.listdir(train_image_path)
                if f.endswith('.png')
            ]
        )
        train_root_masks = sorted(
            [
                os.path.join(train_mask_path, f)
                for f in os.listdir(train_mask_path)
                if f.endswith('.tif')
            ]
        )

        total_train_images = len(train_images) // 121
        val_ratio = 0.2
        num_val_images = int(round(total_train_images * val_ratio))

        print(f"Total training images: {total_train_images}")
        print(f"Number of validation images: {num_val_images}")
        print(
            f"Number of actual training images: {total_train_images - num_val_images}"
        )

        final_train_images = []
        final_train_root_masks = []
        val_images = []
        val_root_masks = []

        for i in range(total_train_images):
            for j in range(121):
                if i < num_val_images:
                    val_images.append(train_images[i * 121 + j])
                    val_root_masks.append(train_root_masks[i * 121 + j])
                else:
                    final_train_images.append(train_images[i * 121 + j])
                    final_train_root_masks.append(
                        train_root_masks[i * 121 + j]
                    )

        self.move_files(
            val_images,
            val_root_masks,
            os.path.join(self.data_patched_path, "val/images"),
            os.path.join(self.data_patched_path, "val/masks"),
        )

    def move_files(
        self, image_list, root_mask_list, target_image_dir, target_mask_dir
    ):
        os.makedirs(target_image_dir, exist_ok=True)
        os.makedirs(target_mask_dir, exist_ok=True)
        for img, mask in tqdm(
            zip(image_list, root_mask_list),
            total=len(image_list),
            desc=f"Preparing {target_image_dir}",
        ):
            shutil.move(
                img, os.path.join(target_image_dir, os.path.basename(img))
            )
            shutil.move(
                mask, os.path.join(target_mask_dir, os.path.basename(mask))
            )

    def upload_to_datastore(self):
        datastore = Datastore(self.workspace, name='workspaceblobstore')

        datastore.upload(
            src_dir="masks",
            target_path='all_images_patches/images',
            overwrite=True,
            show_progress=True,
        )
        datastore.upload(
            src_dir="images",
            target_path='all_masks_patches/masks',
            overwrite=True,
            show_progress=True,
        )

    def register_datasets(self):
        datastore = Datastore(self.workspace, name='workspaceblobstore')

        images_train_set = Dataset.File.from_files(
            path=(datastore, 'all_images_patches')
        )
        masks_train_set = Dataset.File.from_files(
            path=(datastore, 'all_masks_patches')
        )
        #
        # val_set = Dataset.File.from_files(path=(datastore, 'data_patched_all_corrected/val'))
        # test_set = Dataset.File.from_files(path=(datastore, 'data_patched_all_corrected/test'))

        images_train_set.register(
            workspace=self.workspace,
            name='images_simona',
            description='Dataset images patches for ImageDataGen',
            create_new_version=True,
        )
        masks_train_set.register(
            workspace=self.workspace,
            name='masks_simona',
            description="Dataset masks patches for ImageDataGen",
            create_new_version=True,
        )

        # val_set.register(workspace=self.workspace, name='patched_val', description='validation data', create_new_version=True)
        # test_set.register(workspace=self.workspace, name='patched_test', description='test data', create_new_version=True)

    def run(self):
        # self.prepare_patches()
        # self.split_train_validation()
        # self.upload_to_datastore()
        self.register_datasets()


if __name__ == "__main__":
    # Set system path/base directory
    sys.path.append(
        r'C:/Users/Cedri/OneDrive/Documents/GitHub/2023-24d-fai2-adsai-group-cv5'
    )

    # Initialize and run the uploader
    # uploader = AzureDataUploader(
    #     subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
    #     resource_group="buas-y2",
    #     workspace_name="cv5",
    #     base_path=base_folder,
    #     auth=InteractiveLoginAuthentication()
    # )

    # uploader.run()
