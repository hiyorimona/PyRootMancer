
Landmark Extraction Class
=========================

This module defines a class `LandmarkExtraction` for extracting landmarks from images using morphological operations and coordinate filtering.

Initialization
---------------

**Description:** Initializes the LandmarkExtraction class.

.. code-block:: python

   from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

   landmarks = LandmarkExtraction()

Morphological Operations
------------------------

**Description:** Applies morphological operations to remove noise from the input image.
Performs erosion, dilation, and closing operations to clean the image.

.. code-block:: python

   from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

   img = cv2.imread('/path/to/image.jpg')
   processed_image = LandmarkExtraction().opening_closing(img)

Removing Small Components
-------------------------

**Description:** Removes small connected components from the mask based on number of labels.

.. code-block:: python

   from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

   mask = cv2.imread('/path/to/mask.jpg', 0)
   cleaned_mask = LandmarkExtraction().remove_small_components(mask)

Bottom Coordinates Extraction
------------------------------

**Description:** Processes and filters coordinates from a skeleton image dataframe.

.. code-block:: python

   from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

   input_folder = '/path/to/images'
   num_img = 0
   x_coords, y_coords = LandmarkExtraction().get_bottom_coordinates(input_folder, num_img)

Root Detection and Visualization
---------------------------------

**Description:** Detects roots in images and visualizes the detected roots.

.. code-block:: python

   from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

   chosen_images = 'Root Detection'
   input_folder = '/path/to/cleaned/masks'
   test_folder = '/path/to/test/images'
   num_img = 0

   landmarks = LandmarkExtraction()
   detected_image = landmarks.detect(chosen_images, input_folder, test_folder, num_img)

Function Examples
-----------------

.. rst:directive:: py:function:: pyrootmancer.scripts.features.landmark_extraction.LandmarkExtraction.opening_closing(img)

   Example usage:

   .. code-block:: python

      from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

      img = cv2.imread('/path/to/image.jpg')
      processed_image = LandmarkExtraction().opening_closing(img)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.landmark_extraction.LandmarkExtraction.remove_small_components(mask)

   Example usage:

   .. code-block:: python

      from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

      mask = cv2.imread('/path/to/mask.jpg', 0)
      cleaned_mask = LandmarkExtraction().remove_small_components(mask)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.landmark_extraction.LandmarkExtraction.get_bottom_coordinates(input_folder, num_img, threshold=50)

   Example usage:

   .. code-block:: python

      from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

      input_folder = '/path/to/images'
      num_img = 0
      x_coords, y_coords = LandmarkExtraction().get_bottom_coordinates(input_folder, num_img)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.landmark_extraction.LandmarkExtraction.detect(chosen_images, input_folder, test_folder, num_img)

   Example usage:

   .. code-block:: python

      from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

      chosen_images = 'Root Detection'
      input_folder = '/path/to/cleaned/masks'
      test_folder = '/path/to/test/images'
      num_img = 0

      landmarks = LandmarkExtraction()
      detected_image = landmarks.detect(chosen_images, input_folder, test_folder, num_img)

Troubleshooting
---------------

**Issue:** No roots detected in the image.

**Solution:** Check if the input mask image (`mask`) has valid root segments. Adjust parameters in `remove_small_components` and `get_bottom_coordinates` methods for better detection.

.. code-block:: python

   from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

   input_folder = '/path/to/images'
   num_img = 0

   try:
       x_coords, y_coords = LandmarkExtraction().get_bottom_coordinates(input_folder, num_img)
       if not x_coords or not y_coords:
           print("No roots detected.")
   except Exception as e:
       print(f"Error: {e}")

---

**Issue:** Error in image processing functions.

**Solution:** Ensure that `input_folder` and `test_folder` paths are correct and contain valid images. Verify the format and integrity of input images and masks.

.. code-block:: python

   from pyrootmancer.scripts.features.landmark_extraction import LandmarkExtraction

   chosen_images = 'Root Detection'
   input_folder = '/incorrect/path/to/masks'
   test_folder = '/path/to/test/images'
   num_img = 0

   try:
       landmarks = LandmarkExtraction()
       detected_image = landmarks.detect(chosen_images, input_folder, test_folder, num_img)
   except Exception as e:
       print(f"Error: {e}")

Execution
---------

.. code-block:: python

   if __name__ == "__main__":
       landmarks = LandmarkExtraction()
       image_num = 1
       landmarks.detect(
           'Root Detection',
           folder_config.get("data_predictions_clean"),
           folder_config.get("test_folder"),
           image_num
       )

       y, x = landmarks.get_bottom_coordinates(folder_config.get("data_predictions_clean"), image_num)
       logging.info(f"coordinates x: {x}" f" \ncoordinates y: {y}")

