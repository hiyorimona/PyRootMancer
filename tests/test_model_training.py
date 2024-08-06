import logging
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

from prc.models.model_training import ModelTraining
from prc.utils.configuration import folder_config


class TestModelTrain(unittest.TestCase):

    @patch('prc.models.model_training.load_model')
    def test_load_model(self, mock_load_model):
        modelling = ModelTraining()
        models_folder = folder_config.get("models_folder")
        model_name = 'best_model_root_masks'

        # Create a mock return value
        mock_return_value = {'a': 1, 'b': 2}
        mock_load_model.return_value = mock_return_value

        # Test the load_model function
        loaded_model = modelling.load_model(models_folder, model_name)

        self.assertEqual(loaded_model, mock_return_value)
        self.assertTrue(mock_load_model.called)

    @patch.object(os.path, 'join')
    def test_load_model_os_error(self, mock_join):
        modelling = ModelTraining()
        models_folder = folder_config.get("models_folder")
        model_name = 'best_model_root_masks'

        # Raise an error when os.path.join is called
        mock_join.side_effect = FileNotFoundError

        with self.assertRaises(FileNotFoundError):
            modelling.load_model(models_folder, model_name)

    @patch('prc.models.model_training.ImageDataGenerator')
    def test_data_generator(self, mock_image_data_generator):
        mock_datagen = MagicMock()
        mock_image_data_generator.return_value = mock_datagen
        
        # Mock the flow_from_directory to return file paths
        mock_flow = MagicMock()
        mock_flow.filepaths = ['dummy_path']
        mock_datagen.flow_from_directory.return_value = mock_flow

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            image_folder = os.path.join(temp_dir, "images")
            mask_folder = os.path.join(temp_dir, "masks")
            os.makedirs(image_folder)
            os.makedirs(mask_folder)
            
            # Add some sample images
            with open(os.path.join(image_folder, "image1.png"), 'w'):
                pass
            with open(os.path.join(mask_folder, "mask1.png"), 'w'):
                pass
            
            modelling = ModelTraining()
            train_generator, validation_generator, train_image_generator, validation_image_generator = modelling.data_generator(image_folder, mask_folder)

            self.assertIsNotNone(train_generator)
            self.assertIsNotNone(validation_generator)
            self.assertIsNotNone(train_image_generator)
            self.assertIsNotNone(validation_image_generator)

    @patch('prc.models.model_training.ImageDataGenerator')
    def test_data_generator_no_files(self, mock_image_data_generator):
        mock_datagen = MagicMock()
        mock_image_data_generator.return_value = mock_datagen

        mock_datagen.flow_from_directory.return_value.filepaths = []

        modelling = ModelTraining()
        train_generator, validation_generator, train_image_generator, validation_image_generator = modelling.data_generator("dummy_image_folder", "dummy_mask_folder")

        self.assertIsNone(train_generator)
        self.assertIsNone(validation_generator)
        self.assertIsNone(train_image_generator)
        self.assertIsNone(validation_image_generator)

    def test_custom_data_generator(self):
        modelling = ModelTraining()

        mock_image_generator = MagicMock()
        mock_mask_generator = MagicMock()

        mock_image_batch = MagicMock()
        mock_mask_batch = MagicMock()

        mock_image_generator.__next__.return_value = mock_image_batch
        mock_mask_generator.__next__.return_value = mock_mask_batch

        mock_image_batch.shape = (1, 2, 3)
        mock_mask_batch.shape = (1, 2, 3)

        generator = modelling.custom_data_generator(mock_image_generator, mock_mask_generator)

        image_batch, mask_batch = next(generator)
        self.assertEqual(image_batch, mock_image_batch)
        self.assertEqual(mask_batch, mock_mask_batch)

    @patch('prc.models.model_training.ModelCheckpoint')
    @patch('prc.models.model_training.EarlyStopping')
    @patch('prc.models.model_training.ModelTraining.data_generator')
    @patch('prc.models.model_training.unet_model')
    def test_training(self, mock_unet_model, mock_data_generator, mock_early_stopping, mock_model_checkpoint):
        mock_model = MagicMock()
        mock_unet_model.return_value = mock_model

        mock_train_gen = MagicMock()
        mock_val_gen = MagicMock()
        mock_train_image_gen = MagicMock()
        mock_val_image_gen = MagicMock()

        mock_data_generator.return_value = (mock_train_gen, mock_val_gen, mock_train_image_gen, mock_val_image_gen)
        mock_train_image_gen.__len__.return_value = 10
        mock_val_image_gen.samples = 160

        modelling = ModelTraining()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_folder = os.path.join(temp_dir, "models")
            os.makedirs(model_folder)

            trained_model = modelling.training(2, "dummy_image_folder", "dummy_mask_folder", model_folder, "model_name")

            self.assertIsNotNone(trained_model)
            self.assertTrue(mock_unet_model.called)
            self.assertTrue(mock_data_generator.called)
            self.assertTrue(mock_model_checkpoint.called)
            self.assertTrue(mock_early_stopping.called)
            self.assertTrue(mock_model.fit.called)

    @patch('prc.models.model_training.ModelTraining.load_model')
    @patch('prc.models.model_training.cv2.imread')
    @patch('prc.models.model_training.cv2.imwrite')
    @patch('prc.models.model_training.DataPipelineSetup.padder')
    @patch('prc.models.model_training.patchify')
    @patch('prc.models.model_training.unpatchify')
    def test_predict_image(self, mock_unpatchify, mock_patchify, mock_padder, mock_imwrite, mock_imread, mock_load_model):
        modelling = ModelTraining()

        mock_image = np.ones((3000, 3000, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        mock_padded_image = MagicMock()
        mock_padder.return_value = mock_padded_image
        mock_patches = MagicMock()
        mock_patchify.return_value = mock_patches
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_preds = MagicMock()
        mock_model.predict.return_value = mock_preds
        mock_unpatchify.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_folder = temp_dir
            models_folder = temp_dir
            image_path = os.path.join(temp_dir, "image.png")
            with open(image_path, 'w'):
                pass
            model_name = "model"

            predicted_mask = modelling.predict_image(image_path, output_folder, models_folder, model_name)

            self.assertIsNotNone(predicted_mask)
            self.assertTrue(mock_imread.called)
            self.assertTrue(mock_padder.called)
            self.assertTrue(mock_patchify.called)
            self.assertTrue(mock_load_model.called)
            self.assertTrue(mock_model.predict.called)
            self.assertTrue(mock_unpatchify.called)
            self.assertTrue(mock_imwrite.called)

    @patch('prc.models.model_training.ModelTraining.predict_image')
    def test_predict_folder(self, mock_predict_image):
        modelling = ModelTraining()

        mock_predict_image.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_folder = os.path.join(temp_dir, "input")
            output_folder = os.path.join(temp_dir, "output")
            models_folder = os.path.join(temp_dir, "models")

            os.makedirs(input_folder)
            os.makedirs(output_folder)
            os.makedirs(models_folder)

            for i in range(5):
                with open(os.path.join(input_folder, f"image_{i}.png"), 'w'):
                    pass

            modelling.predict_folder(input_folder, output_folder, models_folder, "model_name")

            self.assertTrue(mock_predict_image.called)
            self.assertEqual(mock_predict_image.call_count, 5)


if __name__ == '__main__':
    unittest.main()
