import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from src.features.instance_segmentation import InstanceSegmentation


class TestInstanceSegmentation(unittest.TestCase):
    """
    Unit test class for InstanceSegmentation.

    @autor: Cédric Verhaegh
    """

    @patch('src.features.instance_segmentation.DataPipelineSetup')
    def setUp(self, MockDataPipelineSetup: MagicMock) -> None:
        """
        Set up the test case environment.

        This method is called before every test. It mocks the DataPipelineSetup class
        and initializes an InstanceSegmentation object with the mocked processor.

        Parameters:
        -----------
        MockDataPipelineSetup : MagicMock
            Mocked DataPipelineSetup class.
        """
        # Mock the DataPipelineSetup class
        self.mock_processor = MockDataPipelineSetup.return_value
        # Initialize the InstanceSegmentation object
        self.segmentation = InstanceSegmentation()

    def test_opening_closing(self) -> None:
        """
        Test the opening_closing method of InstanceSegmentation.

        This test verifies that the opening_closing method processes the image correctly
        by checking the shape and content of the returned image.
        """
        # Create a test image of ones
        img = np.ones((10, 10), dtype="uint8") * 255
        # Process the image using the opening_closing method
        processed_img = self.segmentation.opening_closing(img)
        # Check if the shape of the processed image matches the original image
        self.assertEqual(processed_img.shape, img.shape)
        # Check if the content of the processed image matches the original image
        self.assertTrue(np.array_equal(processed_img, img))

    @patch('cv2.imread')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('src.models.model_training.ModelTraining.predict_image')
    def test_test_overlaying(self, mock_predict_image: MagicMock, mock_destroyAllWindows: MagicMock, mock_waitKey: MagicMock,
                             mock_imshow: MagicMock, mock_imread: MagicMock) -> None:
        """
        Test the test_overlaying method of InstanceSegmentation.

        This test verifies that the test_overlaying method correctly overlays a mask on an image,
        processes it, and calls the appropriate functions.

        Parameters:
        -----------
        mock_predict_image : MagicMock
            Mocked predict_image method of ModelTraining.
        mock_destroyAllWindows : MagicMock
            Mocked cv2.destroyAllWindows function.
        mock_waitKey : MagicMock
            Mocked cv2.waitKey function.
        mock_imshow : MagicMock
            Mocked cv2.imshow function.
        mock_imread : MagicMock
            Mocked cv2.imread function.
        """
        # Mock the predict_image method to return an array of ones
        mock_predict_image.return_value = np.ones((3000, 3000), dtype="uint8")
        # Mock the imread method to return an array of ones
        mock_imread.return_value = np.ones((3006, 4202), dtype="uint8") * 255

        # Define paths for the image, output folder, model folder, and model name
        image_path = "path/to/test/image.png"
        output_folder = "path/to/output/folder"
        model_folder = "path/to/model/folder"
        model_name = "model_name"

        # Test the overlaying method
        blended_img = self.segmentation.test_overlaying(image_path, output_folder, model_folder, model_name)

        # Check if the blended image is not None
        self.assertIsNotNone(blended_img)
        # Check if the shape of the blended image matches the expected shape
        self.assertEqual(blended_img.shape, (3006, 4202, 3))
        # Check if imshow was called once
        mock_imshow.assert_called_once()
        # Check if waitKey was called once
        mock_waitKey.assert_called_once()
        # Check if destroyAllWindows was called once
        mock_destroyAllWindows.assert_called_once()

    @patch('cv2.normalize')
    @patch('cv2.addWeighted')
    @patch('skimage.transform.resize')
    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_return_original_size_image(
        self,
        mock_imwrite: MagicMock,
        mock_imread: MagicMock,
        mock_resize: MagicMock,
        mock_addweighted: MagicMock,
        mock_normalize: MagicMock,
    ) -> None:
        """
        Test the return_original_size_image method of InstanceSegmentation.

        This test verifies that the return_original_size_image method correctly processes and resizes images,
        overlays the resized masks, and writes the processed images to disk.

        Parameters:
        -----------
        mock_imwrite : MagicMock
            Mocked cv2.imwrite function.
        mock_imread : MagicMock
            Mocked cv2.imread function.
        mock_resize : MagicMock
            Mocked skimage.transform.resize function.
        mock_addweighted : MagicMock
            Mocked cv2.addWeighted function.
        mock_normalize : MagicMock
            Mocked cv2.normalize function.
        """
        # Mock the imread method to return an array of ones
        mock_imread.return_value = np.ones((3006, 4202), dtype="uint8") * 255
        # Mock the resize method to return an array of ones
        mock_resize.return_value = np.ones((2816, 2816), dtype="uint8") * 255
        # Mock the addWeighted method to return an array of ones
        mock_addweighted.return_value = (
            np.ones((2816, 2816), dtype="uint8") * 255
        )
        # Mock the normalize method to return an array of ones
        mock_normalize.return_value = (
            np.ones((3006, 4202), dtype="uint8") * 255
        )

        # Define paths for the image and output folder
        image_path = "path/to/test/image.png"
        output_folder = "path/to/output/folder"

        # Test the return_original_size_image method
        result = self.segmentation.return_original_size_image(
            image_path, output_folder
        )

        # Check if the result is not None
        self.assertIsNotNone(result)
        # Check if imwrite was called once
        self.assertEqual(mock_imwrite.call_count, 1)

    @patch(
        'src.features.instance_segmentation.DataPipelineSetup.create_folders'
    )
    @patch('os.listdir')
    @patch(
        'src.features.instance_segmentation.InstanceSegmentation.return_original_size_image'
    )
    def test_return_original_size_folder(
        self,
        mock_return_original_size_image: MagicMock,
        mock_listdir: MagicMock,
        mock_create_folders: MagicMock,
    ) -> None:
        """
        Test the return_original_size_folder method of InstanceSegmentation.

        This test verifies that the return_original_size_folder method correctly processes a folder of images,
        resizes the predicted masks, and calls the appropriate functions.

        Parameters:
        -----------
        mock_return_original_size_image : MagicMock
            Mocked return_original_size_image method.
        mock_listdir : MagicMock
            Mocked os.listdir function.
        mock_create_folders : MagicMock
            Mocked create_folders method of DataPipelineSetup.
        """
        # Mock the listdir method to return a list of image names
        mock_listdir.return_value = ['image1.png', 'image2.png']

        # Define paths for the test folder and output folder
        test_folder = "path/to/test/folder"
        output_folder = "path/to/output/folder"

        # Test the return_original_size_folder method
        self.segmentation.return_original_size_folder(
            test_folder, output_folder
        )

        # Check if return_original_size_image was called twice
        self.assertEqual(mock_return_original_size_image.call_count, 2)
        # Check if create_folders was called once
        mock_create_folders.assert_called_once()


if __name__ == '__main__':
    unittest.main()
