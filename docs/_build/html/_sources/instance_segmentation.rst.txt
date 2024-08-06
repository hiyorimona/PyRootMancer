
Instance Segmentation Class
===========================

This module defines a class `InstanceSegmentation` for performing instance segmentation tasks, including overlaying masks on images and resizing predicted masks.

Initialization
---------------

**Description:** Initializes the InstanceSegmentation class.

.. code-block:: python

   from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

   instance_segmentation = InstanceSegmentation()

Overlaying Predicted Mask on Image
-----------------------------------

**Description:** Overlays a predicted mask on a randomly selected original image and displays the result.

.. code-block:: python

   from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

   image_path = '/path/to/original/image.jpg'
   output_folder = '/path/to/output'
   model_folder = '/path/to/model'
   model_name = 'model_name'

   blended_image = InstanceSegmentation().test_overlaying(image_path, output_folder, model_folder, model_name)

Resizing Predicted Masks
-------------------------

**Description:** Resizes predicted masks to match the size of the original images and overlays them.

.. code-block:: python

   from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

   image_path = '/path/to/predicted/mask.jpg'
   output_folder = '/path/to/output'

   resized_image = InstanceSegmentation().return_original_size_image(image_path, output_folder)

Overlaying Masks on Multiple Images
------------------------------------

**Description:** Overlays predicted masks on multiple images from specified folders.

.. code-block:: python

   from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

   test_folder = '/path/to/test/images'
   predicted_folder = '/path/to/predicted/masks'
   output_folder = '/path/to/output'

   InstanceSegmentation().overlay(test_folder, predicted_folder, output_folder)

Resizing Masks for Folder of Images
------------------------------------

**Description:** Resizes predicted masks to match the size of original images and overlays them for all images in a folder.

.. code-block:: python

   from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

   test_folder = '/path/to/test/images'
   output_folder = '/path/to/output'

   InstanceSegmentation().return_original_size_folder(test_folder, output_folder)

Function Examples
-----------------

.. rst:directive:: py:function:: pyrootmancer.scripts.features.instance_segmentation.InstanceSegmentation.test_overlaying(image_path, output_folder, model_folder, model_name)

   Example usage:

   .. code-block:: python

      from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

      image_path = '/path/to/original/image.jpg'
      output_folder = '/path/to/output'
      model_folder = '/path/to/model'
      model_name = 'model_name'

      blended_image = InstanceSegmentation().test_overlaying(image_path, output_folder, model_folder, model_name)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.instance_segmentation.InstanceSegmentation.return_original_size_image(image_path, output_folder)

   Example usage:

   .. code-block:: python

      from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

      image_path = '/path/to/predicted/mask.jpg'
      output_folder = '/path/to/output'

      resized_image = InstanceSegmentation().return_original_size_image(image_path, output_folder)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.instance_segmentation.InstanceSegmentation.overlay(test_folder, predicted_folder, output_folder)

   Example usage:

   .. code-block:: python

      from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

      test_folder = '/path/to/test/images'
      predicted_folder = '/path/to/predicted/masks'
      output_folder = '/path/to/output'

      InstanceSegmentation().overlay(test_folder, predicted_folder, output_folder)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.instance_segmentation.InstanceSegmentation.return_original_size_folder(test_folder, output_folder)

   Example usage:

   .. code-block:: python

      from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

      test_folder = '/path/to/test/images'
      output_folder = '/path/to/output'

      InstanceSegmentation().return_original_size_folder(test_folder, output_folder)

Troubleshooting
---------------

**Issue:** No mask overlay displayed on the image.

**Solution:** Ensure that `image_path`, `output_folder`, `model_folder`, and `model_name` are correctly specified and accessible. Verify that the model predictions are successful and the image paths are valid.

.. code-block:: python

   from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

   image_path = '/path/to/original/image.jpg'
   output_folder = '/path/to/output'
   model_folder = '/path/to/model'
   model_name = 'model_name'

   try:
       blended_image = InstanceSegmentation().test_overlaying(image_path, output_folder, model_folder, model_name)
       if blended_image is None:
           print("No mask overlay displayed.")
   except Exception as e:
       print(f"Error: {e}")

---

**Issue:** Error in resizing predicted masks.

**Solution:** Ensure that `image_path` and `output_folder` paths are correct and accessible. Verify the format and integrity of the predicted mask images.

.. code-block:: python

   from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation

   image_path = '/path/to/predicted/mask.jpg'
   output_folder = '/path/to/output'

   try:
       resized_image = InstanceSegmentation().return_original_size_image(image_path, output_folder)
       if resized_image is None:
           print("Error in resizing predicted masks.")
   except Exception as e:
       print(f"Error: {e}")

Execution
---------

.. code-block:: python

   if __name__ == "__main__":
       instance_segmentation = InstanceSegmentation()

       image_path = '/path/to/original/image.jpg'
       output_folder = '/path/to/output'
       model_folder = '/path/to/model'
       model_name = 'model_name'

       blended_image = instance_segmentation.test_overlaying(image_path, output_folder, model_folder, model_name)
       logging.info("Mask overlay completed.")

       test_folder = '/path/to/test/images'
       predicted_folder = '/path/to/predicted/masks'

       instance_segmentation.overlay(test_folder, predicted_folder, output_folder)
       logging.info("Overlay process completed for multiple images.")

       instance_segmentation.return_original_size_folder(test_folder, output_folder)
       logging.info("Resizing masks completed for folder of images.")

