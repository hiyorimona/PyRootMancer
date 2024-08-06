import argparse
import sys

sys.path.append('E:/Github/2023-24d-fai2-adsai-group-cv5/')
from data.data_preprocessing import DataPipelineSetup


def main(unzip_files, run):
    processor = DataPipelineSetup()
    processor.create_folders()
    if unzip_files:
        processor.unzip("train")
        processor.unzip("test")
        processor.unzip("masks")
        processor.crop(processor.images_folder)
        processor.crop(processor.root_folder)
        processor.crop(processor.shoot_folder)
        processor.save_patches()
    if run:
        processor.crop(processor.images_folder)
        processor.crop(processor.root_folder)
        processor.crop(processor.shoot_folder)
        processor.save_patches()


if __name__ == "__main__":
    description = '''
    This script is used for data ingestion and preprocessing. It sets up a data pipeline, validates the data, 
    trains a model, performs instance segmentation, and extracts landmarks. 

    The main function initializes the data pipeline, model training, data validation, instance segmentation, 
    and landmark extraction. It then creates necessary folders and optionally unzips train, test, and masks files.

    Options:
    --unzip: Unzip, preprocess images and save patches
    --run: Preprocess images and save patches
    '''
    print(description)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--unzip',
        dest='unzip_files',
        action='store_true',
        help='Unzip images and masks files and save patches',
    )
    parser.add_argument(
        '--run',
        dest='run',
        action='store_true',
        help='Preprocess images and save patches',
    )
    parser.set_defaults(unzip_files=False, run=False)
    args = parser.parse_args()
    main(args.unzip_files, args.run)
