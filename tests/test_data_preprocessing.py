import unittest
from unittest.mock import patch, MagicMock, call
import cv2
import numpy as np
import os
import shutil

from prc.utils.configuration import folder_config, param_config
from prc.data.data_preprocessing import DataPipelineSetup


class TestDataPipelineSetup(unittest.TestCase):

    @patch('prc.data.data_preprocessing.os.makedirs')
    def test_create_folders(self, mock_makedirs):
        processor = DataPipelineSetup()
        processor.create_folders()
        expected_calls = [
            call(folder, exist_ok=True) for folder in folder_config.values()
        ]
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)

    @patch('prc.data.data_preprocessing.os.scandir')
    @patch('prc.data.data_preprocessing.ZipFile')
    @patch('prc.data.data_preprocessing.logging.warning')
    @patch('prc.data.data_preprocessing.logging.info')
    def test_unzip_existing_files(self, mock_info, mock_warning, mock_zipfile, mock_scandir):
        processor = DataPipelineSetup()
        mock_scandir.return_value = [MagicMock(is_file=MagicMock(return_value=True))]
        processor.unzip(keyword="train")
        mock_warning.assert_called_once_with(
            'Files already exist in images. Aborting unzip operation to prevent overwriting.'
        )
        mock_zipfile.assert_not_called()
        mock_info.assert_not_called()

    @patch('prc.data.data_preprocessing.os.rename')
    @patch('prc.data.data_preprocessing.shutil.rmtree')
    @patch('prc.data.data_preprocessing.os.scandir', return_value=[])
    @patch('prc.data.data_preprocessing.os.walk', return_value=[('root', [], ['file1.jpg', 'file2.jpg'])])
    @patch('prc.data.data_preprocessing.ZipFile')
    @patch('prc.data.data_preprocessing.logging.info')
    def test_unzip_train(self, mock_info, mock_zipfile, mock_walk, mock_scandir, mock_rmtree, mock_rename):
        processor = DataPipelineSetup()
        with patch('prc.data.data_preprocessing.os.makedirs'):
            processor.unzip(keyword="train")
            mock_zipfile.assert_called_once_with(
                os.path.join(folder_config.get("raw_data_folder"), "train.zip"), 'r'
            )
            mock_zipfile_instance = mock_zipfile.return_value.__enter__.return_value
            mock_zipfile_instance.extractall.assert_called_once_with("train")
            expected_info_calls = [
                call('train file unzipping...'),
                call('2 png images in train zip folder'),
                call('Done!'),
            ]
            mock_info.assert_has_calls(expected_info_calls)

    @patch('prc.data.data_preprocessing.os.rename')
    @patch('prc.data.data_preprocessing.shutil.rmtree')
    @patch('prc.data.data_preprocessing.os.scandir', return_value=[])
    @patch('prc.data.data_preprocessing.os.walk', return_value=[('root', [], ['root_mask1.tiff', 'shoot_mask1.tif'])])
    @patch('prc.data.data_preprocessing.ZipFile')
    @patch('prc.data.data_preprocessing.logging.info')
    def test_unzip_masks(self, mock_info, mock_zipfile, mock_walk, mock_scandir, mock_rmtree, mock_rename):
        processor = DataPipelineSetup()
        with patch('prc.data.data_preprocessing.os.makedirs'):
            processor.unzip(keyword="masks")
            mock_zipfile.assert_called_once_with(
                os.path.join(folder_config.get("raw_data_folder"), "masks.zip"), 'r'
            )
            mock_zipfile_instance = mock_zipfile.return_value.__enter__.return_value
            mock_zipfile_instance.extractall.assert_called_once_with("masks")
            expected_info_calls = [
                call('masks file unzipping...'),
                call('1 tiff images class root in masks zip folder'),
                call('1 tiff images class shoot in masks zip folder'),
                call('Done!'),
            ]
            mock_info.assert_has_calls(expected_info_calls)

    @patch('prc.data.data_preprocessing.os.rename')
    @patch('prc.data.data_preprocessing.shutil.rmtree')
    @patch('prc.data.data_preprocessing.os.scandir', return_value=[])
    @patch('prc.data.data_preprocessing.os.walk', return_value=[('root', [], ['file1.jpg', 'file2.jpg'])])
    @patch('prc.data.data_preprocessing.ZipFile')
    @patch('prc.data.data_preprocessing.logging.info')
    def test_unzip_test(self, mock_info, mock_zipfile, mock_walk, mock_scandir, mock_rmtree, mock_rename):
        processor = DataPipelineSetup()
        with patch('prc.data.data_preprocessing.os.makedirs'):
            processor.unzip(keyword="test")
            mock_zipfile.assert_called_once_with(
                os.path.join(folder_config.get("raw_data_folder"), "test.zip"), 'r'
            )
            mock_zipfile_instance = mock_zipfile.return_value.__enter__.return_value
            mock_zipfile_instance.extractall.assert_called_once_with("test")
            expected_info_calls = [
                call('test file unzipping...'),
                call('2 png images in test zip folder'),
                call('Done!'),
            ]
            mock_info.assert_has_calls(expected_info_calls)
            mock_rmtree.assert_called_once_with("test")

    @patch('prc.data.data_preprocessing.os.listdir')
    @patch('prc.data.data_preprocessing.cv2.imread')
    @patch('prc.data.data_preprocessing.cv2.imwrite')
    @patch('prc.data.data_preprocessing.tq.tqdm')
    @patch('prc.data.data_preprocessing.logging.error')
    @patch('prc.data.data_preprocessing.logging.info')
    def test_crop(self, mock_info, mock_error, mock_tqdm, mock_imwrite, mock_imread, mock_listdir):
        processor = DataPipelineSetup()
        folder = folder_config.get("test_folder")

        # Set up mock return values and side effects
        test_image_1 = '030_43-2-ROOT1-2023-08-08_pvdCherry_OD001_Col0_05-Fish Eye Corrected.png'
        test_image_2 = '030_43-19-ROOT1-2023-08-08_pvdCherry_OD001_Col0_04-Fish Eye Corrected.png'

        mock_listdir.return_value = [test_image_1, test_image_2]
        mock_imread.side_effect = [
            np.random.randint(0, 256, (3000, 3000, 3), dtype=np.uint8),  # for image1.png
            np.random.randint(0, 256, (3000, 3000), dtype=np.uint8),  # for image2.tif
        ]
        mock_tqdm.return_value = enumerate(mock_listdir.return_value)

        # Call the crop method
        processor.crop(folder)

        # Assert listdir was called with the correct folder
        mock_listdir.assert_called_once_with(folder)

        # Assert imread was called with the correct file paths
        expected_image_1_path = os.path.join(folder, test_image_1)
        expected_image_2_path = os.path.join(folder, test_image_2)
        self.assertEqual(mock_imread.call_args_list[0][0][0], expected_image_1_path)
        self.assertEqual(mock_imread.call_args_list[1][0][0], expected_image_2_path)

        # Assert imwrite was called with the correct file paths and processed images
        mock_imwrite.assert_called()

    @patch('prc.data.data_preprocessing.cv2.copyMakeBorder')
    def test_padder(self, mock_copyMakeBorder):
        processor = DataPipelineSetup()
        patch_size = param_config.get("patch_size")
        # Create a test image with dimensions that are not divisible by patch_size
        test_image = np.random.randint(0, 256, (1000, 750, 3), dtype=np.uint8)

        # Expected padding calculations
        h, w = test_image.shape[:2]
        height_padding = ((h // patch_size) + 1) * patch_size - h
        width_padding = ((w // patch_size) + 1) * patch_size - w

        top_padding = height_padding // 2
        bottom_padding = height_padding - top_padding
        left_padding = width_padding // 2
        right_padding = width_padding - left_padding

        # Create the expected padded image using cv2.copyMakeBorder
        expected_padded_image = cv2.copyMakeBorder(
            test_image,
            top_padding,
            bottom_padding,
            left_padding,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        # Set the mock return value for copyMakeBorder
        mock_copyMakeBorder.return_value = expected_padded_image

        # Call the padder function
        padded_image = processor.padder(test_image)

        self.assertTrue(np.array_equal(padded_image, expected_padded_image))

    @patch('prc.data.data_preprocessing.os.listdir')
    @patch('prc.data.data_preprocessing.cv2.imread')
    @patch('prc.data.data_preprocessing.cv2.imwrite')
    @patch('prc.data.data_preprocessing.patchify')
    @patch('prc.data.data_preprocessing.DataPipelineSetup.padder')
    def test_img_patchify(self, mock_padder, mock_patchify, mock_imwrite, mock_imread, mock_listdir):
        # Setup mock return values
        mock_listdir.return_value = ['image1.tif', 'image2.png']
        mock_imread.return_value = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        mock_padder.return_value = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mock_patchify.return_value = np.random.randint(0, 256, (4, 4, 256, 256, 3), dtype=np.uint8)

        # Create an instance of DataPipelineSetup
        processor = DataPipelineSetup()

        # Call the img_patchify method
        img_dir = '/path/to/images'
        save_dir = '/path/to/save'
        img, tifs = processor.img_patchify(img_dir, save_dir)

        # Assertions
        self.assertNotEqual(len(img), 0)
        self.assertNotEqual(len(tifs), 0)
        mock_imwrite.assert_called()

if __name__ == '__main__':
    unittest.main()