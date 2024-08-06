import os


def get_project_root(repository_name="2023-24d-fai2-adsai-group-cv5"):
    """
    root directory of the project based on a known folder name in the path.
    """
    return os.path.join(os.sep.join(os.getcwd().split(os.sep)[: os.getcwd().split(os.sep).index(repository_name) + 1]))


base_folder = get_project_root()
# model_checkpoint = ModelCheckpoint(os.path.join(model_folder, f'{model_name}.keras'), save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')


folder_config = {
    # Data folders - raw, processed and external
    "external_data_folder": os.path.join(base_folder, "src", "src", "data", "external"),
    "raw_data_folder": os.path.join(base_folder, "src", "data", "raw"),
    "processed_data_folder": os.path.join(base_folder, "src", "data", "processed"),
    # Raw data - unpatched - raw images and masks
    "data_unpatched": os.path.join(base_folder, "src", "data", "raw", "data_unpatched"),
    "images_folder_unpatched": os.path.join(base_folder, "src", "data", "raw", "data_unpatched", "images"),
    "test_folder": os.path.join(base_folder, "src", "data", "raw", "data_unpatched", "test"),
    "root_folder_unpatched": os.path.join(base_folder, "src", "data", "raw", "data_unpatched", "root_masks"),
    "shoot_folder_unpatched": os.path.join(base_folder, "src", "data", "raw", "data_unpatched", "shoot_masks"),
    # Raw data - patched - patches of the images and masks
    # Patches of the images and masks for use with ImageDataGenerator()
    "data_patched": os.path.join(base_folder, "src", "data", "raw", "data_patched"),
    "images_folder_patched": os.path.join(base_folder, "src", "data", "raw", "data_patched", "images", "images"),
    "root_folder_patched": os.path.join(base_folder, "src", "data", "raw", "data_patched", "root_masks", "root_masks"),
    "shoot_folder_patched": os.path.join(
        base_folder, "src", "data", "raw", "data_patched", "shoot_masks", "shoot_masks"
    ),
    # Processed predictions - predicted masks from the raw/data_unpatched/test_folder
    "data_predictions": os.path.join(base_folder, "src", "data", "processed", "data_predicted"),
    "data_predictions_clean": os.path.join(base_folder, "src", "data", "processed", "data_predictions_clean"),
    # Models folder
    "models_folder": os.path.join(base_folder, "src", "models"),
}


param_config = {
    "patch_size": 256,
    "input_shape": (256, 256, 3),
    "num_classes": 1,
    "batch_size": 16,
    "epochs": 1,
    # "callbacks": [early_stopping, model_checkpoint]
}
